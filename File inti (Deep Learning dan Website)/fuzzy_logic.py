import numpy as np

class FuzzyKekeringan:
    def __init__(self):
        # Rentang semesta pembicaraan (Universe of Discourse) untuk Input (0-1) dan Output (0-100)
        self.x_range = np.linspace(0, 1, 100)
        self.y_range = np.linspace(0, 100, 100)

    # 1. FUNGSI KEANGGOTAAN (Triangular Membership Function / Kurva Segitiga)
    def trimf(self, x, abc):
        a, b, c = abc
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x)
            y = np.where((x > a) & (x <= b), (x - a) / (b - a), y)
            y = np.where((x > b) & (x < c), (c - x) / (c - b), y)
            return y
        else:
            if x <= a or x >= c: return 0.0
            elif a < x <= b: return (x - a) / (b - a)
            elif b < x < c: return (c - x) / (c - b)
            return 0.0

    # 2. FUZZIFIKASI (Menghitung derajat keanggotaan)
    def fuzzify(self, crisp_inputs):
        suhu, kelembapan, hujan, cahaya, angin = crisp_inputs

        # Parameter kurva segitiga rentang 0-1
        params = {
            'Rendah': [-0.5, 0.0, 0.5],
            'Sedang': [0.1, 0.5, 0.9],
            'Tinggi': [0.5, 1.0, 1.5]
        }

        fuzzy_vals = {
            'Suhu': {k: self.trimf(suhu, v) for k, v in params.items()},
            'Kelembapan': {k: self.trimf(kelembapan, v) for k, v in params.items()},
            'Hujan': {k: self.trimf(hujan, v) for k, v in params.items()},
            'Cahaya': {k: self.trimf(cahaya, v) for k, v in params.items()},
            'Angin': {k: self.trimf(angin, v) for k, v in params.items()}
        }
        return fuzzy_vals

    # 3. RULE BASE & INFERENSI (15 Rule)
    def evaluate_rules(self, f_vals):
        rules = []
        # Rule Ekstrem Kering
        rules.append(('Tinggi', min(f_vals['Suhu']['Tinggi'], f_vals['Hujan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Cahaya']['Tinggi'], f_vals['Kelembapan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Suhu']['Tinggi'], f_vals['Angin']['Tinggi'])))
        rules.append(('Tinggi', min(f_vals['Hujan']['Rendah'], f_vals['Kelembapan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Cahaya']['Tinggi'], f_vals['Hujan']['Rendah'])))

        # Rule Normal
        rules.append(('Sedang', min(f_vals['Suhu']['Sedang'], f_vals['Hujan']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Kelembapan']['Sedang'], f_vals['Angin']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Cahaya']['Sedang'], f_vals['Suhu']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Hujan']['Sedang'], f_vals['Kelembapan']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Angin']['Sedang'], f_vals['Cahaya']['Sedang'])))

        # Rule Basah
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'], f_vals['Suhu']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Kelembapan']['Tinggi'], f_vals['Cahaya']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'], f_vals['Angin']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Suhu']['Rendah'], f_vals['Kelembapan']['Tinggi'])))
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'], f_vals['Cahaya']['Rendah'])))

        agg = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
        for label, val in rules:
            if val > agg[label]:
                agg[label] = val
        return agg

    # 4a. DEFUZZIFIKASI MAMDANI
    def defuzzify_mamdani(self, agg):
        out_params = {'Rendah': [-50, 0, 50], 'Sedang': [10, 50, 90], 'Tinggi': [50, 100, 150]}
        numerator = 0
        denominator = 0
        for y in self.y_range:
            mu_y = max(
                min(agg['Rendah'], self.trimf(y, out_params['Rendah'])),
                min(agg['Sedang'], self.trimf(y, out_params['Sedang'])),
                min(agg['Tinggi'], self.trimf(y, out_params['Tinggi']))
            )
            numerator += y * mu_y
            denominator += mu_y
        return numerator / denominator if denominator != 0 else 0

    # 4b. DEFUZZIFIKASI SUGENO
    def defuzzify_sugeno(self, agg):
        sugeno_const = {'Rendah': 20, 'Sedang': 50, 'Tinggi': 80}
        numerator = 0
        denominator = 0
        for label in ['Rendah', 'Sedang', 'Tinggi']:
            numerator += agg[label] * sugeno_const[label]
            denominator += agg[label]
        return numerator / denominator if denominator != 0 else 0
