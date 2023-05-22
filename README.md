<h1 align='center'>
  Paper Abstract Categorizer
  <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Markdownify" width='120'>
  </a>
</h1>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Setup](#setup)
    * [Windows](#windows)
    * [MacOS](#macos)
* [Instructions](#instructions)
    * [Running a model operation](#running-a-model-operation)
    * [Running the Django application](#running-the-django-application)
* [Contributing](#contributing)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

This project represents a very basic sceleton of an web application that also features ML capabilities. Due to time limitations, we've decided to not proceed with the perfect approach here - fully decoupling the ML framework and App code and only symlinking the tested part. The `ml_framework` folder contains a `core` and `zoo` directories - for ML production code and experimentations respectively. The `storage` folder solely contains the training data (missing from GH since it is too big), intermediate model training checkpoints and preprocessed data. The `app` folder contains the code for a very simple Django application that allows you to categorize your paper abstract into 6 categories using the model trained within the `ml_framework`. 

## Setup

### Windows
```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### MacOS
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### NB!
There's a `abstract_categorizer_model.zip` file that I'll send via email. Please, make sure you unzip it in the storage directory.

## Instructions


### Running a model operation
Running all model operations (hyperparameter tuning, data processing, data exploration and training) is quite straightforward. Each of those operations have a standalone script that one can trigger using:
```
python src/ml_framework/zoo/<NAME_OF_THE_SCRIPT_HERE>.py
```

### Running the Django application
Run the following command to run the application locally (from within `src/app/abstract_categorizer/`):
```
python manage.py runserver
```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- CONTACT -->
## Contact
Hristo Minkov - minkov.h@gmail.com

Codebase Link: [https://github.com/icaka98/iris-ai-model-framework](https://github.com/icaka98/iris-ai-model-framework)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: git_images/present.png