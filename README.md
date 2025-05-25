# Titanic - Survivability Prediction on Titanic using Random Forest Model 🚢

This repository contains a from-scratch implementation of a **Random Forest classifier** to solve the [Kaggle Titanic challenge](https://www.kaggle.com/competitions/titanic/overview).

## 🧠 Overview

The Titanic competition is a classic binary classification task:  
> Predict which passengers survived the Titanic shipwreck using passenger information such as age, sex, class, and ticket fare.

This project does **not use scikit-learn** for modeling — the decision tree and random forest logic are implemented manually to better understand how these algorithms work under the hood.

---

## 🏗️ Project Structure

- `forest.py` – Main script with custom Random Forest implementation
- `train.csv` – Titanic training dataset (from Kaggle)
- `README.md` – Project documentation

---

## 🚀 Getting Started

1. Clone the repository:
  ```bash
  git clone https://github.com/boogen/random-forest.git
  cd random-forest
  ```


2. Install dependencies in a virtual environment
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip3 install -r requirements.txt
  ```
  
3. Run the script:
  ```bash
  python3 forest.py
  ```
---

## 📊 Features Used

The model uses the following features from the Titanic dataset:

- `Pclass` – Ticket class (1st, 2nd, 3rd)
- `Sex` – Gender (encoded as 0/1)
- `Age` – Age of passenger (missing values filled with median)
- `SibSp` – Number of siblings/spouses aboard
- `Parch` – Number of parents/children aboard
- `Fare` – Price paid for the ticket
- `Embarked` – Port of embarkation (S, C, Q; filled with mode and encoded)

---

## 📎 Related Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/overview)
- [Kaggle Titanic Data](https://www.kaggle.com/competitions/titanic/data)
- [Random Forest explanation (Wikipedia)](https://en.wikipedia.org/wiki/Random_forest)
- [Pandas documentation](https://pandas.pydata.org/)
- [NumPy documentation](https://numpy.org/doc/)

---

## 🧑‍💻 Author

Created by Marcin Bugala as a hands-on exercise in building machine learning algorithms from scratch.  
This project is intended for learning and exploration
