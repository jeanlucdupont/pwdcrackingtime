AI-Powered Password Crack Time Estimator. 
Comparing mathematical, AI, and dictionary attack crack time estimation.

Dictionary is the goat.
On super complex password though, AI may have an edge.

<img width="739" height="737" alt="image" src="https://github.com/user-attachments/assets/d74bd385-2b80-4e7e-a743-bf4a24590fa4" />


The script provides both quantitative and qualitative feedback about password strength, entropy, and estimated crack time across different attack scenarios.

Password strength classification uses an ML model trained on the real RockYou breach dataset

**Features**

- Estimation of crack times for: Online attack, Offline (strong hashing), Offline (weak hashing), GPU cluster attack
- Entropy and pattern detection (sequential, repeated, keyboard patterns)
- Optional dictionary attack estimation using zxcvbn
- Input as file, CLI or manual entry
- Model training and persistence with pickle (classifier + regressor + scaler)


**Usage**
```
1. Evaluate a single password
python pwd.py -p MySuperPassword123!

2. Evaluate multiple passwords from a file
python pwd.py -f passwords.txt

3. Evaluate from stdin
cat passwords.txt | python pwd.py --stdin

4. Manual entry
python pwd.py
# Enter passwords one per line (hidden input)
```




**ML training**

Download rockyou (https://weakpass.com/wordlists/rockyou.txt)
```
python rockyou.py
```

Saves:
- password_classifier.pkl
- password_regressor.pkl
- password_scaler.pkl



**Example Output**

<img width="729" height="779" alt="image" src="https://github.com/user-attachments/assets/c24b528e-938d-4477-aa3e-26173d4f65b4" />

<img width="824" height="781" alt="image" src="https://github.com/user-attachments/assets/9d015e9e-6031-42dd-8f28-59a99570a47a" />



**How It Works**
- Feature extraction: Length, variety, entropy, pattern score, bigram ratios, etc.
- ML classification & regression: Predict strength class and log(seconds) to crack.
- Mathematical model: Traditional brute force calculation based on character set.
- Dictionary model: Approximation using zxcvbn (or RockYou fallback).


There is a *strong* possibility I did something wrong in the script. Feel free to correct me.


**Disclaimer**

This tool is for educational and security awareness purposes only.
Do not use it for illegal or unethical activities.
The accuracy of predictions depends on training data and attack assumptions.
