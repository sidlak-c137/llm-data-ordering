# cse517-final-project

## Environment Setup

Finetuning and evaluating the TinyLlama Model on SNLI and GSM8k data requires certain packages. The ```requirements.txt``` file lists what is needed.

For conda environment setup:
```
conda env create -f environment.yml
```

For venv environment setup (on VM) (see [Google Cloud docs](https://cloud.google.com/python/docs/setup#linux)), or anywhere else Python and/or venv might not yet be available:
```
sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```
Then, create venv environment, activate, and install all requirements:
```
cd <project>
python3 -m venv nlp-project

source nlp-project/bin/activate
pip install -r requirements.txt
```

## Finetuning and Evaluating TinyLlama

Scripts for training and evaluating TinyLlama can be found under ```scripts/training```. To set up arguments, create a ```config.json``` file with all training/evaluation parameters:
```
{
    "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "architecture_args": {
        "unfrozen_layers": ["score"],
        "quantized": null
    },
    "dataset": "snli",
    "num_train_samples": 10000,
    "num_val_samples": 1000,
    "num_test_samples": 1000,
    "trainer_args": {
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.01,
        "learning_rate": 4e-4,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 10,
        "batch_size": 8,
        "dataloader_num_workers": 8,
        "load_best_model_at_end": true,
        "strategy": "epoch",
        "output_dir": "experiment_data/"
    },
    "do_train": true
}
```

Then pass the path to this configuration file as the first argument to the ```train.py```:
```
torchrun train.py --config_path="./config.json"
```
Setting ```do_train=False``` will skip training and will only evaluate metrics on the validation and test sets (sampled with ```num_val_samples``` and ```num_test_samples``` from the validation and test samples of the provided dataset name (dataset is loaded from HF)). ```trainer_args``` generally correspond to 



## Getting started
A note a /data:
snli_data_map_coordinates exclude entries where label = -1
snli_ngram_results did not exclude entries where label = -1
Thus difference in length.

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.cs.washington.edu/sidlak/cse517-final-project.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.cs.washington.edu/sidlak/cse517-final-project/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
