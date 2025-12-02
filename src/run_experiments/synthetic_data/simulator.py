import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

def cloglog_inv(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    eta_c = np.clip(eta, -40.0, 20.0)  
    exp_eta = np.exp(eta_c)
    mu = -np.expm1(-exp_eta) # 1 - exp(-x) = -expm1(-x)
    return np.clip(mu, 0.0, 1.0)


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

class SynthParams:
    # Time / RNG
    T: int = 240 # horizon in months (20 years)
    seed: int = 7

    # Baseline demographics
    p_male: float = 0.48
    p_low_edu: float = 0.35
    age_min: int = 18
    age_max: int = 85

    # Anthropometry (height cm)
    male_height_mu: float = 175.0
    male_height_sd: float = 7.0
    female_height_mu: float = 162.0
    female_height_sd: float = 6.0

    # Baseline BMI around which weight is drawn
    base_bmi_mu: float = 27.0
    base_bmi_sd: float = 4.0
    base_bmi_age_slope: float = 0.05 # per year from 50
    base_bmi_male_shift: float = 1.2
    base_bmi_lowedu_shift: float = 1.0

    # Waist mean model: WC = a0 + a_w*W - a_h*H + a_m*male + N(0, wc_sd^2)
    wc_a0: float = 20.0
    wc_a_w: float = 1.00
    wc_a_h: float = 0.05
    wc_a_male: float = 6.0
    wc_sd: float = 3.5

    # Thigh circumference (rough scaling)
    thigh_mu: float = 54.0
    thigh_sd: float = 5.0
    thigh_male_shift: float = 2.0

    # Latent mediator persistence (adiposity -> metabolic risk)
    m0: float = -4.0
    m1_adipo: float = 0.8
    m2_persist: float = 0.5
    m3_disease: float = 1.0
    wc_ref: float = 90.0

    # Hazard model (cloglog) coefficients
    gamma0: float = -10.                        # baseline
    gamma1_per_year: float = math.log(2) / 10.0 # doubling ~every 8 years
    beta_male: float = 0.25
    beta_lowedu: float = 0.18
    beta_smk: float = 0.35
    beta_alc_hi: float = 0.10
    theta_M: float = 0.6
    beta_disease: float = 0.8 # extra hazard for illness
    monthly_hazard_cap: float = 0.20

    # Survey weights (positive)
    weight_mean: float = 1.0
    weight_sd_log: float = 0.35

    # Smoking / alcohol prevalences
    p_current_smoker: float = 0.18
    p_heavy_alcohol: float = 0.12

    # Illness / reverse-causation mechanism
    disease_logit_intercept: float = -6.0
    disease_logit_age_beta: float = 0.10       # per 10 years from 50
    disease_logit_bmi_beta: float = 0.20       # per BMI unit above 27
    disease_weight_loss_mean: float = 8.0      # kg
    disease_weight_loss_sd: float = 4.0        # kg
    min_weight_kg: float = 35.0
    max_weight_kg: float = 240.0


class BaselineSampler:
    """
    Provides baseline covariates, either from real NHANES or synthetic.

    For real-baseline mode you should pass a cleaned nhanes.csv that at least has:
        SEQN, RIDAGEYR, RIAGENDR, BMXHT, BMXWT, BMXBMI, BMXWAIST
    plus optional survey-weight, lifestyle, etc.
    """

    def __init__(
        self,
        params: SynthParams,
        rng: np.random.Generator,
        use_real: bool = False,
        real_csv: Optional[Path | str] = None,
    ) -> None:
        self.params = params
        self.rng = rng
        self.use_real = use_real
        self.real_csv = Path(real_csv) if real_csv is not None else None

    def _sample_baseline_synth(self, n: int) -> pd.DataFrame:
        p, r = self.params, self.rng

        age = r.integers(p.age_min, p.age_max + 1, n)
        male01 = r.binomial(1, p.p_male, n)
        lowedu = r.binomial(1, p.p_low_edu, n)

        height = np.where(
            male01 == 1,
            r.normal(p.male_height_mu, p.male_height_sd, n),
            r.normal(p.female_height_mu, p.female_height_sd, n),
        )

        bmi0 = r.normal(
            loc=(
                p.base_bmi_mu
                + p.base_bmi_age_slope * (age - 50)
                + p.base_bmi_male_shift * male01
                + p.base_bmi_lowedu_shift * lowedu
            ),
            scale=p.base_bmi_sd,
            size=n,
        )

        h_m = height / 100.0
        weight0 = np.clip(bmi0 * (h_m ** 2), p.min_weight_kg, p.max_weight_kg)

        # lifestyle
        smk = r.binomial(1, p.p_current_smoker, n)
        alc_hi = r.binomial(1, p.p_heavy_alcohol, n)

        # anthropometrics
        wc0 = (
            p.wc_a0
            + p.wc_a_w * weight0
            - p.wc_a_h * height
            + p.wc_a_male * male01
            + r.normal(0.0, p.wc_sd, n)
        )
        wc0 = np.maximum(50.0, wc0)

        thigh0 = np.maximum(
            35.0, r.normal(p.thigh_mu + p.thigh_male_shift * male01, p.thigh_sd, n)
        )

        # survey weights
        wts = np.exp(r.normal(np.log(p.weight_mean), p.weight_sd_log, n))

        df = pd.DataFrame(
            {
                "SEQN": np.arange(n, dtype=int),
                "RIDAGEYR": age.astype(int),
                "RIAGENDR": np.where(male01 == 1, 1, 2).astype(int),
                "EDU": lowedu.astype(int),
                "BMXHT": height.astype(float),
                "BMXWT_raw": weight0.astype(float),
                "BMXWAIST_raw": wc0.astype(float),
                "BMXTHICR": thigh0.astype(float),
                "SMQ": smk.astype(int),
                "ALQ": alc_hi.astype(int),
                "WTMEC2YR": wts.astype(float),
            }
        )

        # derived indices (raw)
        height_m = df["BMXHT"].to_numpy() / 100.0
        weight_raw = df["BMXWT_raw"].to_numpy()
        waist_raw = df["BMXWAIST_raw"].to_numpy()
        bmi_raw = weight_raw / (height_m ** 2)

        df["BMXBMI_raw"] = bmi_raw

        with np.errstate(divide="ignore", invalid="ignore"):
            absi = waist_raw / (np.power(bmi_raw, 2.0 / 3.0) * np.power(height_m, 0.5))
        df["ABSI_raw"] = np.nan_to_num(absi, nan=np.nan, posinf=np.nan, neginf=np.nan)

        df["ICO_raw"] = waist_raw / np.maximum(df["BMXHT"].to_numpy(), 1e-6)

        wc_m = waist_raw / 100.0
        r_wc = wc_m / (2.0 * math.pi)
        r_h = 0.5 * height_m
        z = 1.0 - np.clip((r_wc ** 2) / (np.maximum(r_h, 1e-9) ** 2), 0.0, 1.0)
        df["BRI_raw"] = 364.2 - 365.5 * np.sqrt(z)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["WTR_raw"] = waist_raw / np.maximum(thigh0, 1e-6)

        return df

    def _sample_baseline_real(self, n: int) -> pd.DataFrame:
        if self.real_csv is None:
            raise ValueError("real_csv must be provided when use_real=True")

        df = pd.read_csv(self.real_csv)

        required = ["RIDAGEYR", "RIAGENDR", "BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Real NHANES file is missing required columns: {missing}. "
                "Adapt BaselineSampler._sample_baseline_real if your schema differs."
            )
        
        df = df.dropna(subset=required).copy()

        df = df.copy()
        df["BMXWT_raw"] = df["BMXWT"]
        df["BMXWAIST_raw"] = df["BMXWAIST"]
        df["BMXBMI_raw"] = df["BMXBMI"]

        if "EDU" not in df.columns:
            if "DMDEDUC2" in df.columns:
                df["EDU"] = (df["DMDEDUC2"].fillna(0.0) <= 2).astype(int)
            else:
                df["EDU"] = 0

        if "SMQ" not in df.columns:
            smq_cols = [c for c in df.columns if c.startswith("SMQ")]
            if smq_cols:
                df["SMQ"] = (df[smq_cols[0]] == 1).astype(int)
            else:
                df["SMQ"] = 0

        if "ALQ" not in df.columns:
            alq_cols = [c for c in df.columns if c.startswith("ALQ")]
            if alq_cols:
                df["ALQ"] = (df[alq_cols[0]] >= 3).astype(int)
            else:
                df["ALQ"] = 0

        if "BMXTHICR" not in df.columns:
            df["BMXTHICR"] = 52.0

        if "WTMEC2YR" not in df.columns:
            df["WTMEC2YR"] = 1.0

        # derived raw indices
        height_m = df["BMXHT"].to_numpy() / 100.0
        weight_raw = df["BMXWT_raw"].to_numpy()
        waist_raw = df["BMXWAIST_raw"].to_numpy()
        bmi_raw = df["BMXBMI_raw"].to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            absi = waist_raw / (np.power(bmi_raw, 2.0 / 3.0) * np.power(height_m, 0.5))
        df["ABSI_raw"] = np.nan_to_num(absi, nan=np.nan, posinf=np.nan, neginf=np.nan)
        df["ICO_raw"] = waist_raw / np.maximum(df["BMXHT"].to_numpy(), 1e-6)

        wc_m = waist_raw / 100.0
        r_wc = wc_m / (2.0 * math.pi)
        r_h = 0.5 * height_m
        z = 1.0 - np.clip((r_wc ** 2) / (np.maximum(r_h, 1e-9) ** 2), 0.0, 1.0)
        df["BRI_raw"] = 364.2 - 365.5 * np.sqrt(z)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["WTR_raw"] = waist_raw / np.maximum(df["BMXTHICR"].to_numpy(), 1e-6)

        if "SEQN" not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": "SEQN"})

        # sample
        if n <= len(df):
            sampled = df.sample(
                n=n,
                replace=False,
                random_state=int(self.rng.integers(0, 2**32 - 1)),
            )
        else:
            sampled = df.sample(
                n=n,
                replace=True,
                random_state=int(self.rng.integers(0, 2**32 - 1)),
            )

        sampled = sampled.reset_index(drop=True)
        sampled["SEQN"] = np.arange(len(sampled), dtype=int)
        return sampled

    def sample_baseline(self, n: int) -> pd.DataFrame:
        if self.use_real:
            return self._sample_baseline_real(n)
        return self._sample_baseline_synth(n)

class NHANESSimulator:
    def __init__(
        self,
        params: Optional[SynthParams] = None,
        use_real_baseline: bool = False,
        real_csv: Optional[Path | str] = None,
    ) -> None:
        self.params = params or SynthParams()
        self.rng = np.random.default_rng(self.params.seed)
        self.baseline_sampler = BaselineSampler(
            self.params, self.rng, use_real=use_real_baseline, real_csv=real_csv
        )

    # mean waist given weight/height/sex
    def wc_mean(
        self,
        weight_kg: np.ndarray,
        height_cm: np.ndarray,
        male01: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        return (
            p.wc_a0
            + p.wc_a_w * weight_kg
            - p.wc_a_h * height_cm
            + p.wc_a_male * male01
        )

    def _augment_with_latent_and_disease(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add disease/latent structure and produce observed anthropometrics.

        Input df is expected to have raw fields:
            BMXWT_raw, BMXWAIST_raw, BMXBMI_raw, BMXHT, RIDAGEYR, RIAGENDR, EDU, SMQ, ALQ
        plus *_raw shape indices from BaselineSampler.
        """
        p, r = self.params, self.rng
        df = df.copy()

        age = df["RIDAGEYR"].to_numpy(dtype=float)
        height_cm = df["BMXHT"].to_numpy(dtype=float)
        height_m = height_cm / 100.0
        weight_raw = df["BMXWT_raw"].to_numpy(dtype=float)
        bmi_raw = df["BMXBMI_raw"].to_numpy(dtype=float)
        waist_raw = df["BMXWAIST_raw"].to_numpy(dtype=float)
        male01 = (df["RIAGENDR"].to_numpy(dtype=int) == 1).astype(int)

        # latent BMI – here matching raw BMI
        bmi_latent = bmi_raw.copy()

        # illness probability depending on latent BMI and age
        age_centered = (age - 50.0) / 10.0
        logit_p = (
            p.disease_logit_intercept
            + p.disease_logit_age_beta * age_centered
            + p.disease_logit_bmi_beta * (bmi_latent - 27.0)
        )
        prob_disease = logistic(logit_p)
        disease = r.binomial(1, prob_disease, size=len(df))

        # disease induced weight loss
        weight_loss = r.normal(
            loc=p.disease_weight_loss_mean,
            scale=p.disease_weight_loss_sd,
            size=len(df),
        )
        weight_loss = np.clip(weight_loss, 0.0, 0.5 * weight_raw)

        weight_obs = np.where(disease == 1, weight_raw - weight_loss, weight_raw)
        weight_obs = np.clip(weight_obs, p.min_weight_kg, p.max_weight_kg)
        bmi_obs = weight_obs / (height_m ** 2)

        # update waist based on new weight
        # keep some dependence on raw WC
        wc_mean_new = self.wc_mean(weight_obs, height_cm, male01)
        wc_noise = self.rng.normal(0.0, p.wc_sd, size=len(df))
        waist_obs = np.maximum(50.0, 0.5 * waist_raw + 0.5 * (wc_mean_new + wc_noise))

        # store latent and observed variables
        df["BMI_latent"] = bmi_latent
        df["disease"] = disease.astype(int)
        df["prob_disease"] = prob_disease

        df["BMXWT"] = weight_obs
        df["BMXBMI"] = bmi_obs
        df["BMXWAIST"] = waist_obs

        # observed shape indices (post-disease)
        with np.errstate(divide="ignore", invalid="ignore"):
            absi = waist_obs / (np.power(bmi_obs, 2.0 / 3.0) * np.power(height_m, 0.5))
        df["ABSI"] = np.nan_to_num(absi, nan=np.nan, posinf=np.nan, neginf=np.nan)
        df["ICO"] = waist_obs / np.maximum(height_cm, 1e-6)

        wc_m = waist_obs / 100.0
        r_wc = wc_m / (2.0 * math.pi)
        r_h = 0.5 * height_m
        z = 1.0 - np.clip((r_wc ** 2) / (np.maximum(r_h, 1e-9) ** 2), 0.0, 1.0)
        df["BRI"] = 364.2 - 365.5 * np.sqrt(z)

        thigh = df["BMXTHICR"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["WTR"] = waist_obs / np.maximum(thigh, 1e-6)

        return df
    
    def _update_M_and_hazard(self, p, age0, t, wc, M_prev, disease, male, lowedu, smk, alc):
        dev = (wc - p.wc_ref) / 10.0
        M = logistic(p.m0 + p.m1_adipo * (dev ** 2)
                            + p.m2_persist * M_prev
                            + p.m3_disease * disease)
        eta = (p.gamma0
            + p.gamma1_per_year * (age0 + t / 12.0)
            + p.beta_male * male
            + p.beta_lowedu * lowedu
            + p.beta_smk * smk
            + p.beta_alc_hi * alc
            + p.theta_M * M
            + p.beta_disease * disease)
        h = cloglog_inv(eta)
        h = np.minimum(h, p.monthly_hazard_cap)
        return M, h


    def simulate_survival(self, base: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate discrete-time survival with adiposity-driven latent mediator.

        Returns base plus:
            - time: months to event or censor (max T)
            - event: 1 if death, 0 if censored
        """
        p = self.params
        df = self._augment_with_latent_and_disease(base)

        n = len(df)
        age0 = df["RIDAGEYR"].to_numpy(dtype=float)
        male = (df["RIAGENDR"].to_numpy(dtype=int) == 1).astype(int)
        lowedu = df["EDU"].to_numpy(dtype=int)
        wc = df["BMXWAIST"].to_numpy(dtype=float)
        smk = df["SMQ"].to_numpy(dtype=int)
        alc = df["ALQ"].to_numpy(dtype=int)
        disease = df["disease"].to_numpy(dtype=int)

        M = np.zeros(n, dtype=float)
        alive = np.ones(n, dtype=bool)
        t_event = np.full(n, p.T, dtype=int)
        died = np.zeros(n, dtype=int)

        for t in range(p.T):
            M, h = self._update_M_and_hazard(
                p, age0, t, wc, M, disease, male, lowedu, smk, alc
            )
            
            u = self.rng.random(n)
            new_events = (u < h) & alive
            if np.any(new_events):
                alive[new_events] = False
                t_event[new_events] = t + 1
                died[new_events] = 1

            if not alive.any():
                break

        df["time"] = t_event.astype(int)
        df["event"] = died.astype(int)
        return df

    def risk_constant_weight(
        self,
        row: pd.Series,
        T_eval: int,
        w_const: float,
    ) -> float:
        """
        Risk of death by T_eval months under an intervention that sets
        weight to w_const for this individual (height fixed).
        """
        p = self.params

        age0 = float(row["RIDAGEYR"])
        male = 1 if int(row["RIAGENDR"]) == 1 else 0
        lowedu = int(row["EDU"])
        smk = int(row["SMQ"])
        alc = int(row["ALQ"])
        disease = int(row.get("disease", 0))
        H = float(row["BMXHT"])

        wc_const = self.wc_mean(
            np.array([w_const], dtype=float),
            np.array([H], dtype=float),
            np.array([male], dtype=int),
        )[0]

        M = 0.0
        logS = 0.0
        for t in range(min(T_eval, p.T)):
            M_arr, h_arr = self._update_M_and_hazard(
                p,
                age0=np.array([age0], dtype=float),
                t=t,
                wc=np.array([wc_const], dtype=float),
                M_prev=np.array([M], dtype=float),
                disease=np.array([disease], dtype=float),
                male=np.array([male], dtype=float),
                lowedu=np.array([lowedu], dtype=float),
                smk=np.array([smk], dtype=float),
                alc=np.array([alc], dtype=float),
            )
            M = float(M_arr[0])
            h = float(h_arr[0])
            h = min(h, p.monthly_hazard_cap)
            logS += math.log(max(1e-12, 1.0 - h))
        return 1.0 - math.exp(logS)

    def sample_dataset(
        self,
        n: int,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify_events: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Return (train_X, y_train, test_X, y_test, full_panel).

        full_panel carries all latent/raw variables (e.g. BMI_latent, disease),
        while train_X / test_X drop obvious leak columns.
        """
        base = self.baseline_sampler.sample_baseline(n)
        panel = self.simulate_survival(base)

        # drop leak columns from features.
        drop_cols = {
            "time",
            "event",
            "BMI_latent",
            "disease",
            "prob_disease",
            "BMXWT_raw",
            "BMXWAIST_raw",
            "BMXBMI_raw",
            "ABSI_raw",
            "ICO_raw",
            "BRI_raw",
            "WTR_raw",
        }
        feature_cols = [
            c for c in panel.columns
            if c not in drop_cols and c not in {"SEQN"}
        ]

        X_all = panel[feature_cols].copy()
        time_all = panel["time"].astype(float).to_numpy()
        event_all = panel["event"].astype(int).astype(bool).to_numpy()
        y_all = Surv.from_arrays(event=event_all, time=time_all)

        stratify = event_all if stratify_events else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return X_train, y_train, X_test, y_test, panel


_GLOBAL_SIMULATOR: Optional[NHANESSimulator] = None
_GLOBAL_PANEL: Optional[pd.DataFrame] = None


def load_nhanes_survival_simulated(
    n: int = 10000,
    time_col: str = "time",
    event_col: str = "event",
    id_col: str = "SEQN",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify_events: bool = True,
    extra_exclude_cols: Optional[list[str]] = None,
    use_real_baseline: bool = False,
    real_csv: Optional[Path | str] = None,
    params: Optional[SynthParams] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Public dataset API, similar to your real-data loader.

    Also caches the full panel and simulator globally so
    simulate_I_ground_truth_counterfactuals can act as a pure function
    of (test_X, T_eval).
    """
    global _GLOBAL_SIMULATOR, _GLOBAL_PANEL

    sim = NHANESSimulator(
        params=params,
        use_real_baseline=use_real_baseline,
        real_csv=real_csv,
    )
    X_train, y_train, X_test, y_test, panel = sim.sample_dataset(
        n=n,
        test_size=test_size,
        random_state=random_state,
        stratify_events=stratify_events,
    )

    _GLOBAL_SIMULATOR = sim
    _GLOBAL_PANEL = panel

    if extra_exclude_cols:
        X_train = X_train.drop(columns=[c for c in extra_exclude_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in extra_exclude_cols if c in X_test.columns])

    return X_train, y_train, X_test, y_test


def _golden_section_min(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-2,
    max_iter: int = 60,
) -> float:
    phi = (1.0 + 5 ** 0.5) / 2.0
    invphi = 1.0 / phi

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    f_c = f(c)
    f_d = f(d)
    it = 0
    while abs(b - a) > tol and it < max_iter:
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - invphi * (b - a)
            f_c = f(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + invphi * (b - a)
            f_d = f(d)
        it += 1
    return 0.5 * (a + b)


def simulate_I_ground_truth_counterfactuals(
    data: pd.DataFrame,
    T_eval: int,
    w_bounds: Tuple[float, float] = (10.0, 200.0),
    return_details: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Ground-truth individual index

        I_i(T) = p_i(w_obs, T) - min_w p_i(w, T)

    using the same hazard and mediator dynamics as the simulator.
    """
    global _GLOBAL_SIMULATOR, _GLOBAL_PANEL

    if _GLOBAL_SIMULATOR is None or _GLOBAL_PANEL is None:
        raise RuntimeError(
            "no global simulator/panel available"
            "call load_nhanes_survival_simulated() first"
        )

    sim = _GLOBAL_SIMULATOR
    panel = _GLOBAL_PANEL

    results = []
    extras: Dict[int, Dict[str, float]] = {}

    for idx in data.index:
        if idx not in panel.index:
            raise KeyError(
                f"index {idx} not found in cached panel"
            )
        row = panel.loc[idx]

        seqn = int(row["SEQN"]) if "SEQN" in row else int(idx)
        w_obs = float(row["BMXWT"])

        p_obs = sim.risk_constant_weight(row, T_eval, w_obs)

        def objective(w: float) -> float:
            w_clipped = float(np.clip(w, w_bounds[0], w_bounds[1]))
            return sim.risk_constant_weight(row, T_eval, w_clipped)

        w_star = _golden_section_min(objective, w_bounds[0], w_bounds[1], tol=1e-1)
        p_star = objective(w_star)

        results.append(
            {
                "SEQN": seqn,
                "p_obs": float(p_obs),
                "w_star": float(w_star),
                "p_star": float(p_star),
                "I_i": float(p_obs - p_star),
            }
        )
        if return_details:
            extras[seqn] = {
                "w_obs": w_obs,
                "BMXWT_raw": float(row.get("BMXWT_raw", np.nan)),
                "BMI_latent": float(row.get("BMI_latent", np.nan)),
                "disease": int(row.get("disease", 0)),
            }

    out = pd.DataFrame(results)
    if return_details:
        return out, extras
    return out


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_nhanes_survival_simulated(
        n=2000,
        # use_real_baseline=False,
        use_real_baseline=True,
        real_csv="data/processed/nhanes.csv",
        random_state=555,
    )
    print("Train shape:", X_tr.shape)
    print("Test shape:", X_te.shape)
    print("y_tr[:3]:", y_tr[:3])

    I_df = simulate_I_ground_truth_counterfactuals(X_te, T_eval=120)
    print(I_df.head())
    print("Mean I_i:", I_df["I_i"].mean())
    
    from matplotlib import pyplot as plt
    plt.hist(I_df['I_i'], bins=30)
    plt.show()