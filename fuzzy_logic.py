import numpy as np

class FuzzyKekeringan:
    def __init__(self):
        # Rentang semesta pembicaraan (Universe of Discourse) untuk Input (0-1) dan Output (0-100)
        self.x_range = np.linspace(0, 1, 200)
        self.y_range = np.linspace(0, 100, 200)

        # Parameter fungsi keanggotaan TRAPESIUM untuk input (0-1)
        # Format: [a, b, c, d] -> naik dari a ke b, datar b ke c, turun c ke d
        self.input_trap_params = {
            'Rendah': [-0.5, -0.1, 0.2, 0.5],
            'Sedang': [0.2, 0.35, 0.65, 0.8],
            'Tinggi': [0.5, 0.8, 1.1, 1.5]
        }

        # Parameter fungsi keanggotaan TRAPESIUM untuk output (0-100)
        self.output_trap_params = {
            'Rendah': [-50, -10, 15, 40],
            'Sedang': [15, 35, 65, 85],
            'Tinggi': [60, 80, 110, 150]
        }

    # 1a. FUNGSI KEANGGOTAAN TRAPESIUM (Trapezoidal Membership Function)
    def trapmf(self, x, abcd):
        a, b, c, d = abcd
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x, dtype=float)
            # Naik
            mask_rise = (x >= a) & (x < b)
            if b != a:
                y = np.where(mask_rise, (x - a) / (b - a), y)
            else:
                y = np.where(mask_rise, 1.0, y)
            # Datar
            y = np.where((x >= b) & (x <= c), 1.0, y)
            # Turun
            mask_fall = (x > c) & (x <= d)
            if d != c:
                y = np.where(mask_fall, (d - x) / (d - c), y)
            else:
                y = np.where(mask_fall, 1.0, y)
            return y
        else:
            if x < a or x > d:
                return 0.0
            elif a <= x < b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b <= x <= c:
                return 1.0
            elif c < x <= d:
                return (d - x) / (d - c) if d != c else 1.0
            return 0.0

    # 1b. FUNGSI KEANGGOTAAN SEGITIGA (tetap dipertahankan untuk keperluan lain)
    def trimf(self, x, abc):
        a, b, c = abc
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x, dtype=float)
            y = np.where((x > a) & (x <= b), (x - a) / (b - a), y)
            y = np.where((x > b) & (x < c), (c - x) / (c - b), y)
            return y
        else:
            if x <= a or x >= c: return 0.0
            elif a < x <= b: return (x - a) / (b - a)
            elif b < x < c: return (c - x) / (c - b)
            return 0.0

    # 2. FUZZIFIKASI menggunakan fungsi keanggotaan TRAPESIUM
    def fuzzify(self, crisp_inputs):
        suhu, kelembapan, hujan, cahaya, angin = crisp_inputs
        params = self.input_trap_params

        fuzzy_vals = {
            'Suhu':       {k: self.trapmf(suhu,       v) for k, v in params.items()},
            'Kelembapan': {k: self.trapmf(kelembapan, v) for k, v in params.items()},
            'Hujan':      {k: self.trapmf(hujan,      v) for k, v in params.items()},
            'Cahaya':     {k: self.trapmf(cahaya,     v) for k, v in params.items()},
            'Angin':      {k: self.trapmf(angin,      v) for k, v in params.items()},
        }
        return fuzzy_vals

    # 3. RULE BASE & INFERENSI (15 Rule)
    def evaluate_rules(self, f_vals):
        rules = []
        # Rule Ekstrem Kering
        rules.append(('Tinggi', min(f_vals['Suhu']['Tinggi'],       f_vals['Hujan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Cahaya']['Tinggi'],     f_vals['Kelembapan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Suhu']['Tinggi'],       f_vals['Angin']['Tinggi'])))
        rules.append(('Tinggi', min(f_vals['Hujan']['Rendah'],      f_vals['Kelembapan']['Rendah'])))
        rules.append(('Tinggi', min(f_vals['Cahaya']['Tinggi'],     f_vals['Hujan']['Rendah'])))

        # Rule Normal
        rules.append(('Sedang', min(f_vals['Suhu']['Sedang'],       f_vals['Hujan']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Kelembapan']['Sedang'], f_vals['Angin']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Cahaya']['Sedang'],     f_vals['Suhu']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Hujan']['Sedang'],      f_vals['Kelembapan']['Sedang'])))
        rules.append(('Sedang', min(f_vals['Angin']['Sedang'],      f_vals['Cahaya']['Sedang'])))

        # Rule Basah
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'],      f_vals['Suhu']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Kelembapan']['Tinggi'], f_vals['Cahaya']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'],      f_vals['Angin']['Rendah'])))
        rules.append(('Rendah', min(f_vals['Suhu']['Rendah'],       f_vals['Kelembapan']['Tinggi'])))
        rules.append(('Rendah', min(f_vals['Hujan']['Tinggi'],      f_vals['Cahaya']['Rendah'])))

        agg = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
        for label, val in rules:
            if val > agg[label]:
                agg[label] = val
        return agg

    # 4a. DEFUZZIFIKASI MAMDANI (menggunakan trapesium output)
    def defuzzify_mamdani(self, agg):
        out_params = self.output_trap_params
        numerator = 0
        denominator = 0
        for y in self.y_range:
            mu_y = max(
                min(agg['Rendah'], self.trapmf(y, out_params['Rendah'])),
                min(agg['Sedang'], self.trapmf(y, out_params['Sedang'])),
                min(agg['Tinggi'], self.trapmf(y, out_params['Tinggi']))
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
