import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput

"""
Modified model wrappers to modify HuggingFace models so loss calculations can be modified.
"""

class AutoModelForSequenceClassificationWithLoss(torch.nn.Module):
    def __init__(self, configs):
        super(AutoModelForSequenceClassificationWithLoss, self).__init__()
        self.configs = configs
        bnb_config = None
        if configs["architecture_args"]["quantized"] is not None:
            bnb_config = BitsAndBytesConfig(
                # load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.float16
                load_in_8bit=True,
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(configs["model_name"], quantization_config=bnb_config, num_labels=3)

    def forward(self, input_ids, attention_mask, labels, hardnesses=None, steps=1):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if hardnesses is None:
            return outputs
        generated_labels = F.log_softmax(outputs.logits, dim=-1)
        return SequenceClassifierOutput(loss=self._compute_loss(generated_labels, labels, hardnesses, steps), logits = outputs.logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        
    def _compute_loss(self, generated_labels, predicted_labels, hardnesses, steps):
        loss_values = -generated_labels[range(predicted_labels.shape[0]), predicted_labels]
        if self.configs["loss_calc"] == "baseline":
            hardnesses = torch.ones_like(hardnesses)
        elif self.configs["loss_calc"] == "triangle":
            hardnesses = ((2 * hardnesses - 1) * (steps - 1)) + 1
        elif self.configs["loss_calc"] == "crazy":
            hardnesses = (1.5 - 5 * (0.5 - hardnesses)) / (steps + 0.2) + 1
        else:
            raise ValueError(f"Loss Calc {self.configs['loss_calc']} unsupported.")
        
        loss = torch.multiply(loss_values, hardnesses).mean()
        # assert torch.isclose(loss, F.cross_entropy(generated_labels, predicted_labels)).item()
        return loss