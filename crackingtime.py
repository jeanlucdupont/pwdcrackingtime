import re
import math
import numpy as np
import pickle
import os
import argparse
import sys
import getpass
from typing                     import Iterable, List
from zxcvbn                     import zxcvbn
from collections                import Counter
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import train_test_split

C_ROCKYOU                                   = "rockyou.txt"

def f_readfile(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]

def f_pwdinput():
    return [line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]

def f_pwdmanualinput():
    pw_list                                 = []
    print("Enter passwords one per line. Leave empty line and press Enter when done.")
    while True:
        pw                                  = getpass.getpass("Password (hidden): ")
        if pw == "":
            break
        pw_list.append(pw)
    return pw_list

class AIPasswordf_passwordcrackeffort:
   
    def __init__(self, rockyou_path=None, use_pretrained=True):
        self.rockyou_path                   = rockyou_path
        self.scaler                         = StandardScaler()
        
        # Try to load pretrained models
        if use_pretrained and self._f_loadmodels():
            print("Loaded pretrained models\n")
        elif rockyou_path and os.path.exists(rockyou_path):
            print(f"Training on RockYou dataset: {rockyou_path}\n\n")
            self._f_rockyoutraining()
        else:
            print("No RockYou file found, using synthetic data (less accurate)\n\n")
            self.strength_classifier        = self._f_strengthclassifier()
            self.crack_time_predictor       = self._f_cracktimepredictor()

    def _f_loadmodels(self):
        try:
            with open('password_classifier.pkl', 'rb') as f:
                self.strength_classifier    = pickle.load(f)
            with open('password_regressor.pkl', 'rb') as f:
                self.crack_time_predictor   = pickle.load(f)
            with open('password_scaler.pkl', 'rb') as f:
                self.scaler                 = pickle.load(f)
            return True
        except:
            return False
    
    def _f_savemodels(self):
        with open('password_classifier.pkl', 'wb') as f:
            pickle.dump(self.strength_classifier, f)
        with open('password_regressor.pkl', 'wb') as f:
            pickle.dump(self.crack_time_predictor, f)
        with open('password_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Models saved to disk")
    
    def _f_rockyoutraining(self):
        print("Loading RockYou passwords...")
        np.random.seed(42)
        passwords                           = self._f_rockyouload()
        print(f"Loaded {len(passwords)} passwords")
        print("Extracting features...")
        
        # Build frequency over passwords, rank and map
        from collections import Counter
        freq                                = Counter(passwords)
        unique_pwds                         = list(freq.keys())
        unique_counts                       = np.array([freq[p] for p in unique_pwds], dtype=float)
        order                               = unique_counts.argsort()[::-1]
        percentiles                         = np.linspace(0.0, 1.0, num=len(unique_pwds), endpoint=False)
        rank_map                            = { unique_pwds[idx]: percentiles[pos]
                                                for pos, idx in enumerate(order) }
        features_list                       = []
        strength_labels                     = []
        crack_time_labels                   = []
        
        for i, pwd in enumerate(passwords):
            if i % 10000 == 0:
                print(f"  Processing {i}/{len(passwords)}...")

            features                        = self.f_passwordfeatures(pwd)
            features_list.append(features)
            rank                            = rank_map[pwd]
                   
            if rank < 0.5:  # Most common 50% = weak
                strength                    = 0
            elif rank < 0.85:  # Next 35% = moderate
                strength                    = 1
            else:  # Top 15% = strong
                strength                    = 2
            strength_labels.append(strength)

            # Crack time estimation 
            if rank < 0.01:     # 1%
                crack_time                  = np.random.uniform(-3, 0)      # milliseconds to 1 second
            elif rank < 0.1:    # 10%
                crack_time                  = np.random.uniform(0, 2)       # 1s to 100s
            elif rank < 0.5:    # 50%
                crack_time                  = np.random.uniform(2, 6)       # 100s to ~11 days
            elif rank < 0.85:   # 85%
                crack_time                  = np.random.uniform(6, 10)      # days to months
            else:               # Top 15%
                crack_time                  = np.random.uniform(10, 15)     # months to years
                        
            # Adjust crack time
            crack_time                      += (features[2] / 50) * 2       # entropy bonus
            crack_time                      -= features[3] * 5              # pattern penalty
            crack_time_labels.append(crack_time)
        
        X                                   = np.array(features_list)
        y_strength                          = np.array(strength_labels)
        y_crack_time                        = np.array(crack_time_labels)     
        self.scaler.fit(X)
        X_scaled                            = self.scaler.transform(X)
        
        print("Training Random Forest Classifier...")
        self.strength_classifier = RandomForestClassifier(
            n_estimators                    = 100,
            max_depth                       = 15,
            min_samples_split               = 10,
            random_state                    = 42,
            n_jobs                          = -1
        )
        self.strength_classifier.fit(X_scaled, y_strength)
        
        print("Training Gradient Boosting Regressor...")
        self.crack_time_predictor = GradientBoostingRegressor(
            n_estimators                    = 200,
            learning_rate                   = 0.1,
            max_depth                       = 6,
            random_state                    = 42
        )
        self.crack_time_predictor.fit(X_scaled, y_crack_time)
        
        print("Training complete (Took its time. I know)")
        self._f_savemodels()
    
    def _f_rockyouload(self, max_passwords=50000):
        passwords                           = []
        
        try:
            with open(self.rockyou_path, 'r', encoding='latin-1', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_passwords:
                        break
                    pwd                     = line.strip()
                    if len(pwd) < 4 or len(pwd) > 30:
                        continue
                    passwords.append(pwd)
        except Exception as e:
            print(f"Error loading RockYou file: {e}")
            raise
        
        return passwords
    
    def _f_strengthclassifier(self):
        model                               = RandomForestClassifier(
            n_estimators                    = 200, 
            max_depth                       = 16, 
            min_samples_split               = 8, 
            random_state                    = 42, 
            n_jobs                          = -1, 
            class_weight="balanced_subsample")
        X_train, y_train                    = self._f_generatesynthetic()
        if not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X_train)
        X_train_scaled                      = self.scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)
        return model    
        
    def _f_cracktimepredictor(self):
        model                               = RandomForestRegressor(
            n_estimators                    = 300, 
            max_depth                       = 18, 
            min_samples_split               = 6, 
            random_state                    = 42, 
            n_jobs                          = -1)
        X_train, y_crack                    = self._f_synthetictime()
        if not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X_train)
        X_train_scaled                      = self.scaler.transform(X_train)
        model.fit(X_train_scaled, y_crack)
        return model
       
    def _f_generatesynthetic(self):
        np.random.seed(42)
        n_samples                           = 5000
        weak_features = np.column_stack([
            np.random.uniform(4,    10,     n_samples//3),      # length (short)
            np.random.uniform(1,    2,      n_samples//3),      # variety (low)
            np.random.uniform(15,   35,     n_samples//3),      # entropy (low)
            np.random.uniform(0.3,  0.9,    n_samples//3),      # pattern score (HIGH - key difference!)
            np.random.uniform(4,    8,      n_samples//3),      # unique chars (low)
            np.random.uniform(0.2,  0.5,    n_samples//3),      # repetition rate
            np.random.uniform(0.5,  1.0,    n_samples//3),      # lowercase ratio (high)
            np.random.uniform(0,    0.1,    n_samples//3),      # uppercase ratio (none/low)
            np.random.uniform(0,    0.4,    n_samples//3),      # digit ratio
            np.random.uniform(0,    0.05,   n_samples//3),      # special ratio (none)
            np.random.uniform(1,    1.5,    n_samples//3),      # position variety (low)
            np.random.uniform(0.05, 0.3,    n_samples//3),      # type changes (few)
            np.random.uniform(0.4,  0.7,    n_samples//3),      # bigram unique ratio
            np.random.uniform(4,    18,     n_samples//3),      # interaction term
            np.random.uniform(2,    4,      n_samples//3),      # normalized entropy
        ])
        
        moderate_features = np.column_stack([
            np.random.uniform(8,    14,     n_samples//3),
            np.random.uniform(3,    4,      n_samples//3),
            np.random.uniform(35,   60,     n_samples//3),
            np.random.uniform(0.05, 0.3,    n_samples//3),  
            np.random.uniform(7,    12,     n_samples//3),
            np.random.uniform(0.05, 0.25,   n_samples//3),
            np.random.uniform(0.2,  0.6,    n_samples//3),
            np.random.uniform(0.1,  0.3,    n_samples//3),
            np.random.uniform(0.1,  0.3,    n_samples//3),
            np.random.uniform(0.05, 0.15,   n_samples//3),
            np.random.uniform(1.3,  2,      n_samples//3),
            np.random.uniform(0.3,  0.6,    n_samples//3),
            np.random.uniform(0.6,  0.85,   n_samples//3),
            np.random.uniform(24,   56,     n_samples//3),
            np.random.uniform(3.5,  5.5,    n_samples//3),
        ])
        
        strong_features = np.column_stack([
            np.random.uniform(14,   24,     n_samples//3),
            np.random.uniform(4,    4,      n_samples//3),       
            np.random.uniform(60,   120,    n_samples//3),
            np.random.uniform(0,    0.1,    n_samples//3),     
            np.random.uniform(12,   24,     n_samples//3),
            np.random.uniform(0,    0.08,   n_samples//3),
            np.random.uniform(0.15, 0.35,   n_samples//3),
            np.random.uniform(0.15, 0.35,   n_samples//3),
            np.random.uniform(0.15, 0.35,   n_samples//3),
            np.random.uniform(0.15, 0.35,   n_samples//3),
            np.random.uniform(1.5,  2,      n_samples//3),
            np.random.uniform(0.7,  0.95,   n_samples//3),  
            np.random.uniform(0.85, 0.98,   n_samples//3), 
            np.random.uniform(56,   96,     n_samples//3),
            np.random.uniform(5,    8,      n_samples//3),
        ])
        
        X                                   = np.vstack([weak_features, moderate_features, strong_features])
        Y                                   = np.array([0] * (n_samples//3) + [1] * (n_samples//3) + [2] * (n_samples//3))
        return X, Y
        
    def f_dictionaryguesses(self, password):
        # Try zxcvbn (best)
        try:
            info                            = zxcvbn(password)
            guesses                         = max(1, int(info.get("guesses", 1)))
            return guesses
        except Exception:
            print(f'No zxcvbn :(')
            pass

        # Try rock you dictionnary
        if hasattr(self, "rank_map") and password in self.rank_map:
            pct                             = self.rank_map[password]  
            guesses                         = int(10 ** ((1.0 - pct) * 6))  
            return max(1, guesses)

        # Fallback to fallback heuristic 
        feats                               = self.f_passwordfeatures(password)
        length                              = feats[0]
        variety                             = feats[1]
        pattern                             = feats[3]
        base_log                            = max(0, 6 - (length - 6) * 0.8 - (variety - 2) * 1.5 - pattern * 3.0)
        guesses                             = int(10 ** max(0.0, base_log))
        return max(1, guesses)

    def f_dictionarytime(self, password, hash_type='offline_fast'):
        guesses = self.f_dictionaryguesses(password)
        speeds = {
            'online':       10.0,
            'offline_slow': 1e3,    
            'offline_fast': 1e9,    
            'gpu_cluster':  1e12
        }
        speed                               = speeds.get(hash_type, speeds['offline_fast'])
        seconds                             = max(1e-9, (guesses / 2.0) / speed)
        return seconds
    
    def _f_synthetictime(self):
        X_train, y_strength                 = self._f_generatesynthetic()
        pattern_scores                      = X_train[:, 3]
        entropy_scores                      = X_train[:, 2]       
        y_crack_time                        = np.zeros(len(y_strength))        
        weak_mask                           = y_strength == 0
        y_crack_time[weak_mask]             = (-2 + pattern_scores[weak_mask] * 4 - (entropy_scores[weak_mask] - 20) / 15)
        moderate_mask                       = y_strength == 1
        y_crack_time[moderate_mask]         = (4 + np.random.uniform(0, 4, np.sum(moderate_mask)) + (1 - pattern_scores[moderate_mask]) * 2 )
        strong_mask                         = y_strength == 2
        y_crack_time[strong_mask]           = (8 + np.random.uniform(0, 7, np.sum(strong_mask)) + (1 - pattern_scores[strong_mask]) * 3)
        y_crack_time                        += np.random.normal(0, 0.3, len(y_crack_time))            # Make some noise!

        return X_train, y_crack_time
    
    def f_passwordfeatures(self, password):
        if not password:
            return np.zeros(15)
        
        length                              = len(password)
        unique_chars                        = len(set(password))
        lowercase_count                     = sum(1 for c in password if c.islower())
        uppercase_count                     = sum(1 for c in password if c.isupper())
        digit_count                         = sum(1 for c in password if c.isdigit())
        special_count                       = sum(1 for c in password if not c.isalnum())
        variety = sum([any(c.islower() for c in password), any(c.isupper() for c in password), any(c.isdigit() for c in password), any(not c.isalnum() for c in password)])
        
        # Entropy calculation
        counter                             = Counter(password)
        entropy                             = -sum((count/length) * math.log2(count/length) 
                                                    for count in counter.values()) * length
       
        # Pattern detection scores (squashed to ~0-1, higher = more patterns)
        sequential_score                    = len(re.findall(r'(012|123|234|345|456|567|678|789|abc|bcd|cde)', password.lower())) / max(length, 1)
        repeated_score                      = len(re.findall(r'(.)\1{1,}', password)) / max(length, 1)
        keyboard_score                      = len(re.findall(r'(qwerty|asdfgh|zxcvbn)', password.lower())) / max(length, 1)
        _pattern_sum                        = sequential_score + repeated_score + keyboard_score
        pattern_score                       = 1.0 - math.exp(-_pattern_sum)   
        
        # Character distribution
        char_repetition_rate                = 1 - (unique_chars / length)
        
        # Locate entropy
        position_variety                    = len(set([password[0], password[-1]])) if length > 1 else 1
        
        # Consecutive character  changes 
        def _ctype(ch):
            if ch.islower():  return 0
            if ch.isupper():  return 1
            if ch.isdigit():  return 2
            return 3
        type_changes                        = 0
        for i in range(len(password) - 1):
            if _ctype(password[i]) != _ctype(password[i+1]):
                type_changes                += 1
        
        # bigram entropy
        bigrams                             = [password[i:i+2] for i in range(len(password)-1)]
        bigram_unique_ratio                 = len(set(bigrams)) / max(len(bigrams), 1)
        
        features                            = np.array([
            length,
            variety,
            entropy,
            pattern_score,  
            unique_chars,
            char_repetition_rate,
            lowercase_count / max(length, 1),
            uppercase_count / max(length, 1),
            digit_count / max(length, 1),
            special_count / max(length, 1),
            position_variety,
            type_changes / max(length - 1, 1),  
            bigram_unique_ratio,
            length * variety,
            entropy / max(length, 1)  
        ])
        
        return features
    
    def f_predictstrenght(self, password):
        features                            = self.f_passwordfeatures(password).reshape(1, -1)
        features_scaled                     = self.scaler.transform(features)
        strength_class                      = self.strength_classifier.predict(features_scaled)[0]
        probabilities                       = self.strength_classifier.predict_proba(features_scaled)[0]        
        return strength_class, probabilities
        
    
    def f_predicttime(self, password, attack_type='offline_fast'):
        features                            = self.f_passwordfeatures(password).reshape(1, -1)
        features_scaled                     = self.scaler.transform(features)
        log_seconds                         = self.crack_time_predictor.predict(features_scaled)[0]
        base_seconds                        = 10 ** log_seconds
        attack_multipliers                  = {'online': 1e-7, 'offline_slow': 1e-1, 'offline_fast': 1.0, 'gpu_cluster': 100.0 }       
        multiplier                          = attack_multipliers.get(attack_type, 1.0)
        adjusted_seconds                    = base_seconds / multiplier
        
        return adjusted_seconds
    
    def f_mathematicaltime(self, password, attack_type='offline_fast'):
        has_lower                           = bool(re.search(r'[a-z]', password))
        has_upper                           = bool(re.search(r'[A-Z]', password))
        has_digit                           = bool(re.search(r'\d', password))
        has_special                         = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password))
        
        charset_size                        = 0
        if has_lower:
            charset_size                    += 26
        if has_upper:
            charset_size                    += 26
        if has_digit:
            charset_size                    += 10
        if has_special:
            charset_size                    += 32
        
        search_space                        = charset_size ** len(password)
        attack_speeds                       = {'online': 10, 'offline_slow': 1e4, 'offline_fast': 1e11, 'gpu_cluster': 1e13 }
        speed                               = attack_speeds.get(attack_type, attack_speeds['offline_fast'])        
        seconds                             = (search_space / 2) / speed
        
        return seconds
    
    def f_time2human(self, seconds):
        if seconds < 1:
            return "< 1 second"
        elif seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.2f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.2f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.2f} days"
        else:
            years = seconds / 31536000
            if years < 1000:
                return f"{years:.2f} years"
            elif years < 1000000:
                return f"{years/1000:.2f} thousand years"
            elif years < 1e9:
                return f"{years/1e6:.2f} million years"
            elif years < 1e12:
                return f"{years/1e9:.2f} billion years"
            else:
                return f"{years/1e12:.2f} trillion years"
    
    def f_passwordcrackeffort(self, password):
        if not password:
            return {"error": "Password cannot be empty"}
        
        strength_class, probabilities       = self.f_predictstrenght(password)
        
        # Extract features for transparency
        features                            = self.f_passwordfeatures(password)
        
        # Predict crack times using ML
        ai_crack_times                      = {}
        for attack in ['online', 'offline_slow', 'offline_fast', 'gpu_cluster']:
            seconds                         = self.f_predicttime(password, attack)
            ai_crack_times[attack]          = self.f_time2human(seconds)
        
        # Calculate mathematical crack times
        math_crack_times                    = {}
        for attack in ['online', 'offline_slow', 'offline_fast', 'gpu_cluster']:
            seconds                         = self.f_mathematicaltime(password, attack)
            math_crack_times[attack]        = self.f_time2human(seconds)
            
        dict_crack_times                    = {}
        for attack in ['online', 'offline_slow', 'offline_fast', 'gpu_cluster']:
            sec                             = self.f_dictionarytime(password, attack)
            dict_crack_times[attack]        = self.f_time2human(sec)            
        
        return {
            'probability_distribution': {
                'Weak':                         f"{probabilities[0]*100:.1f}%",
                'Moderate':                     f"{probabilities[1]*100:.1f}%",
                'Strong':                       f"{probabilities[2]*100:.1f}%"
            },
            'key_features': {
                'Length':                       int(features[0]),
                'Character variety':            int(features[1]),
                'Entropy':                      f"{features[2]:.2f} bits",
                'Pattern score':                f"{features[3]:.3f}",
                'Unique Characters':            int(features[4])
            },
            'ai_crack_times': {
                'Online Attack':                ai_crack_times['online'],
                'Offline (Strong Hashing)':     ai_crack_times['offline_slow'],
                'Offline (Weak Hashing)':       ai_crack_times['offline_fast'],
                'GPU Cluster':                  ai_crack_times['gpu_cluster']
            },
            'mathematical_crack_times': {
                'Online Attack':                math_crack_times['online'],
                'Offline (Strong Hashing)':     math_crack_times['offline_slow'],
                'Offline (Weak Hashing)':       math_crack_times['offline_fast'],
                'GPU Cluster':                  math_crack_times['gpu_cluster']
            },
            'dictionary_crack_times': {
               'Online Attack':                 dict_crack_times['online'],
               'Offline (Strong Hashing)':      dict_crack_times['offline_slow'],
               'Offline (Weak Hashing)':        dict_crack_times['offline_fast'],
               'GPU Cluster':                   dict_crack_times['gpu_cluster']
            }
        }


if __name__ == "__main__":
    print("-" * 80)
    print("Evaluation (AI and then simple mathematics) of the time required\nto brute force a password.")
    print("Aslo evaluating how long it would take with a dictionary attack.")    
    print("Trained on Real RockYou Password Breach Dataset.")
    print("-" * 80)

    parser                                  = argparse.ArgumentParser(description="")
    group                                   = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--password", nargs="+", help="One or more passwords on the command line (less secure).")
    group.add_argument("-f", "--file", help="File containing passwords, one per line.")
    group.add_argument("--stdin", action="store_true", help="Read passwords from stdin (one per line).")
    args                                    = parser.parse_args()

    if args.password:
        passwords                           = args.password
    elif args.file:
        passwords                           = f_readfile(args.file)
    elif args.stdin:
        passwords                           = f_pwdinput()
    else:
        passwords                           = f_pwdmanualinput()

    if not passwords:
        print("No passwords provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    f_passwordcrackeffort                   = AIPasswordf_passwordcrackeffort(rockyou_path=C_ROCKYOU, use_pretrained=True)
    
    for pwd in passwords:
        print("_" * 80)
        print(f"Analyzing: {pwd}")
        result                              = f_passwordcrackeffort.f_passwordcrackeffort(pwd)
            
        print(f"\nPROBABILITY DISTRIBUTION:")
        for strength, prob in result['probability_distribution'].items():
            bar_length                      = int(float(prob.rstrip('%')) / 5)
            bar                             = '█' * bar_length
            print(f"   {strength:10s}: {bar:20s} {prob}")
        
        print(f"\nEXTRACTED FEATURES")
        for feature, value in result['key_features'].items():
            print(f"   • {feature:<31}{value}")
        
        print(f"\nAI-PREDICTED CRACK TIMES")
        for attack, time in result['ai_crack_times'].items():
            print(f"   • {attack:<31}{time}")
        
        print(f"\nMATHEMATICAL (Brute Force) CRACK TIMES")
        for attack, time in result['mathematical_crack_times'].items():
            print(f"   • {attack:<31}:{time}")

        print(f"\nDICTIONARY ATTACK CRACK TIMES")
        for attack, time in result['dictionary_crack_times'].items():
            print(f"   • {attack:<31}{time}")
    
