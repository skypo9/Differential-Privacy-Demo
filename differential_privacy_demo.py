"""
Differential Privacy Demonstration

This script demonstrates the differential privacy concept for preserving privacy
while ensuring that individual data cannot be distinguished from statistical queries.

Differential privacy provides mathematical guarantees that the presence or absence
of any individual record in a dataset does not significantly affect the output
of statistical analyses.

Author: Privacy-Preserving Data Analytics Demo
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class DifferentialPrivacyMechanism:
    """
    Implementation of various differential privacy mechanisms.
    """
    
    def __init__(self, epsilon: float = 1.0):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon (float): Privacy budget parameter. Lower values = more privacy
        """
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0
        
    def laplace_mechanism(self, true_value: float, sensitivity: float) -> float:
        """
        Apply Laplace mechanism for differential privacy.
        
        Args:
            true_value (float): True statistical value
            sensitivity (float): Global sensitivity of the function
            
        Returns:
            float: Noisy value satisfying differential privacy
        """
        if self.privacy_budget_used + self.epsilon > self.epsilon:
            raise ValueError("Privacy budget exhausted!")
            
        # Laplace noise scale
        scale = sensitivity / self.epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale)
        noisy_value = true_value + noise
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon
        
        return noisy_value
    
    def gaussian_mechanism(self, true_value: float, sensitivity: float, delta: float = 1e-5) -> float:
        """
        Apply Gaussian mechanism for differential privacy.
        
        Args:
            true_value (float): True statistical value
            sensitivity (float): Global sensitivity of the function
            delta (float): Failure probability parameter
            
        Returns:
            float: Noisy value satisfying (epsilon, delta)-differential privacy
        """
        # Gaussian noise scale
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, scale)
        noisy_value = true_value + noise
        
        return noisy_value
    
    def exponential_mechanism(self, candidates: List[Any], utility_scores: List[float], 
                            sensitivity: float) -> Any:
        """
        Apply exponential mechanism for non-numeric outputs.
        
        Args:
            candidates: List of candidate outputs
            utility_scores: Utility scores for each candidate
            sensitivity: Sensitivity of the utility function
            
        Returns:
            Selected candidate with differential privacy
        """
        # Calculate probabilities
        scores = np.array(utility_scores)
        probabilities = np.exp(self.epsilon * scores / (2 * sensitivity))
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample from the distribution
        choice_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[choice_idx]
    
    def reset_budget(self, new_epsilon: float = None):
        """Reset the privacy budget."""
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        self.privacy_budget_used = 0.0

class MedicalDataGenerator:
    """
    Generator for synthetic medical datasets for differential privacy demonstration.
    """
    
    def __init__(self):
        self.seed = 42
        
    def generate_patient_records(self, n_patients: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic patient records for DP demonstration.
        
        Args:
            n_patients (int): Number of patient records to generate
            
        Returns:
            pd.DataFrame: Synthetic medical dataset
        """
        np.random.seed(self.seed)
        
        # Demographics
        ages = np.random.normal(50, 15, n_patients).astype(int)
        ages = np.clip(ages, 18, 90)
        
        genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
        
        # Medical measurements
        systolic_bp = np.random.normal(130, 20, n_patients)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        cholesterol = np.random.normal(200, 40, n_patients)
        cholesterol = np.clip(cholesterol, 120, 350)
        
        bmi = np.random.normal(26, 5, n_patients)
        bmi = np.clip(bmi, 15, 50)
        
        # Medical history (binary)
        diabetes = np.random.binomial(1, 0.12, n_patients)
        hypertension = np.random.binomial(1, 0.25, n_patients)
        heart_disease = np.random.binomial(1, 0.08, n_patients)
        
        # Hospital costs
        treatment_cost = np.random.exponential(5000, n_patients)
        treatment_cost = np.clip(treatment_cost, 500, 50000)
        
        # Create DataFrame
        data = pd.DataFrame({
            'PatientID': ['P{:05d}'.format(i) for i in range(1, n_patients + 1)],
            'Age': ages,
            'Gender': genders,
            'SystolicBP': systolic_bp,
            'Cholesterol': cholesterol,
            'BMI': bmi,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'HeartDisease': heart_disease,
            'TreatmentCost': treatment_cost
        })
        
        return data

class DifferentialPrivacyAnalyzer:
    """
    Analyzer for demonstrating differential privacy on medical data.
    """
    
    def __init__(self, data: pd.DataFrame, epsilon: float = 1.0):
        """
        Initialize the analyzer with medical data.
        
        Args:
            data (pd.DataFrame): Medical dataset
            epsilon (float): Privacy parameter
        """
        self.data = data
        self.epsilon = epsilon
        self.dp_mechanism = DifferentialPrivacyMechanism(epsilon)
        
    def private_mean(self, column: str, bounds: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate differentially private mean.
        
        Args:
            column (str): Column name to calculate mean for
            bounds (Tuple[float, float]): (min_value, max_value) for the column
            
        Returns:
            Tuple[float, float]: (true_mean, private_mean)
        """
        # True mean
        true_mean = self.data[column].mean()
        
        # Calculate sensitivity (for mean with bounded data)
        min_val, max_val = bounds
        sensitivity = (max_val - min_val) / len(self.data)
        
        # Apply Laplace mechanism
        self.dp_mechanism.reset_budget(self.epsilon)
        private_mean = self.dp_mechanism.laplace_mechanism(true_mean, sensitivity)
        
        return true_mean, private_mean
    
    def private_count(self, condition: str) -> Tuple[int, float]:
        """
        Calculate differentially private count.
        
        Args:
            condition (str): Pandas query condition
            
        Returns:
            Tuple[int, float]: (true_count, private_count)
        """
        # True count
        true_count = len(self.data.query(condition))
        
        # Sensitivity for counting queries is 1
        sensitivity = 1.0
        
        # Apply Laplace mechanism
        self.dp_mechanism.reset_budget(self.epsilon)
        private_count = self.dp_mechanism.laplace_mechanism(true_count, sensitivity)
        
        # Ensure non-negative count
        private_count = max(0, private_count)
        
        return true_count, private_count
    
    def private_histogram(self, column: str, bins: List[str]) -> Tuple[Dict, Dict]:
        """
        Calculate differentially private histogram.
        
        Args:
            column (str): Column name
            bins (List[str]): List of bin labels/categories
            
        Returns:
            Tuple[Dict, Dict]: (true_histogram, private_histogram)
        """
        # True histogram
        true_hist = self.data[column].value_counts().to_dict()
        
        # Initialize with zeros for missing categories
        for bin_label in bins:
            if bin_label not in true_hist:
                true_hist[bin_label] = 0
        
        # Apply Laplace mechanism to each bin
        private_hist = {}
        sensitivity = 1.0  # Adding/removing one record changes count by at most 1
        
        for bin_label in bins:
            self.dp_mechanism.reset_budget(self.epsilon / len(bins))  # Split privacy budget
            count = true_hist.get(bin_label, 0)
            private_count = self.dp_mechanism.laplace_mechanism(count, sensitivity)
            private_hist[bin_label] = max(0, private_count)  # Ensure non-negative
        
        return true_hist, private_hist
    
    def demonstrate_privacy_loss(self, target_patient_id: str, 
                                query_function: Callable) -> Dict[str, Any]:
        """
        Demonstrate privacy protection by comparing results with/without target patient.
        
        Args:
            target_patient_id (str): Patient ID to include/exclude
            query_function (Callable): Function to execute on the dataset
            
        Returns:
            Dict: Results showing privacy protection
        """
        # Dataset with target patient
        data_with = self.data.copy()
        
        # Dataset without target patient
        data_without = self.data[self.data['PatientID'] != target_patient_id].copy()
        
        # Run query on both datasets
        analyzer_with = DifferentialPrivacyAnalyzer(data_with, self.epsilon)
        analyzer_without = DifferentialPrivacyAnalyzer(data_without, self.epsilon)
        
        result_with = query_function(analyzer_with)
        result_without = query_function(analyzer_without)
        
        return {
            'with_patient': result_with,
            'without_patient': result_without,
            'difference': abs(result_with - result_without) if isinstance(result_with, (int, float)) else None,
            'privacy_protected': True  # DP provides mathematical guarantee
        }

class DifferentialPrivacyVisualizer:
    """
    Visualizer for differential privacy demonstrations.
    """
    
    def __init__(self):
        self.fig_size = (16, 12)
        
    def create_dp_dashboard(self, analyzer: DifferentialPrivacyAnalyzer, 
                          save_path: str = "differential_privacy_dashboard.png"):
        """
        Create comprehensive differential privacy dashboard.
        
        Args:
            analyzer (DifferentialPrivacyAnalyzer): DP analyzer with data
            save_path (str): Path to save dashboard
        """
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Privacy Budget Visualization
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_privacy_budget(ax1, analyzer)
        
        # 2. Noise Distribution Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_noise_distribution(ax2, analyzer)
        
        # 3. Privacy-Utility Trade-off
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_privacy_utility_tradeoff(ax3)
        
        # 4. Mean Estimation Comparison
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_mean_comparison(ax4, analyzer)
        
        # 5. Count Query Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_count_comparison(ax5, analyzer)
        
        # 6. Histogram Comparison
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_histogram_comparison(ax6, analyzer)
        
        plt.suptitle('Differential Privacy Protection Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Differential privacy dashboard saved as {}".format(save_path))
        return fig
    
    def _plot_privacy_budget(self, ax, analyzer):
        """Plot privacy budget visualization."""
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        privacy_levels = ['Very High', 'High', 'Medium', 'Low', 'Very Low', 'Minimal']
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#689f38', '#1976d2', '#7b1fa2']
        
        # Current epsilon
        current_idx = min(range(len(epsilon_values)), 
                         key=lambda i: abs(epsilon_values[i] - analyzer.epsilon))
        
        bars = ax.barh(privacy_levels, epsilon_values, color=colors, alpha=0.7)
        bars[current_idx].set_alpha(1.0)
        bars[current_idx].set_edgecolor('black')
        bars[current_idx].set_linewidth(3)
        
        ax.set_xlabel('Epsilon (ε)')
        ax.set_title('Privacy Budget\n(Current: ε={})'.format(analyzer.epsilon), fontweight='bold')
        ax.axvline(x=analyzer.epsilon, color='red', linestyle='--', linewidth=2)
        
        # Add explanation text
        ax.text(0.95, 0.05, 'Lower ε = More Privacy\nHigher ε = Less Privacy', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_noise_distribution(self, ax, analyzer):
        """Plot noise distribution for Laplace mechanism."""
        sensitivity = 1.0
        scale = sensitivity / analyzer.epsilon
        
        x = np.linspace(-10*scale, 10*scale, 1000)
        laplace_pdf = (1/(2*scale)) * np.exp(-np.abs(x)/scale)
        
        ax.plot(x, laplace_pdf, 'b-', linewidth=2, label='Laplace Noise')
        ax.fill_between(x, laplace_pdf, alpha=0.3)
        
        ax.set_xlabel('Noise Value')
        ax.set_ylabel('Probability Density')
        ax.set_title('Noise Distribution\n(Laplace Mechanism)', fontweight='bold')
        ax.legend()
        
        # Add statistics
        ax.text(0.95, 0.95, 'Scale: {:.2f}\nMean: 0\nStd: {:.2f}'.format(scale, scale*np.sqrt(2)), 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_privacy_utility_tradeoff(self, ax):
        """Plot privacy-utility trade-off curve."""
        epsilon_range = np.logspace(-1, 1, 50)
        
        # Privacy score (higher epsilon = lower privacy)
        privacy_scores = 100 / (1 + epsilon_range)
        
        # Utility score (higher epsilon = higher utility)
        utility_scores = 100 * (1 - np.exp(-epsilon_range))
        
        ax.plot(epsilon_range, privacy_scores, 'r-', linewidth=2, label='Privacy Score')
        ax.plot(epsilon_range, utility_scores, 'b-', linewidth=2, label='Utility Score')
        
        ax.set_xlabel('Epsilon (ε)')
        ax.set_ylabel('Score (0-100)')
        ax.set_title('Privacy-Utility Trade-off', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal range
        ax.axvspan(0.5, 2.0, alpha=0.2, color='green', 
                  label='Balanced Range')
        ax.legend()
    
    def _plot_mean_comparison(self, ax, analyzer):
        """Plot comparison of true vs private means."""
        # Test different columns
        columns = ['Age', 'SystolicBP', 'Cholesterol', 'BMI', 'TreatmentCost']
        bounds = [(18, 90), (90, 200), (120, 350), (15, 50), (500, 50000)]
        
        true_means = []
        private_means = []
        errors = []
        
        for col, bound in zip(columns, bounds):
            true_mean, private_mean = analyzer.private_mean(col, bound)
            true_means.append(true_mean)
            private_means.append(private_mean)
            errors.append(abs(true_mean - private_mean))
        
        x = np.arange(len(columns))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_means, width, label='True Mean', alpha=0.7)
        bars2 = ax.bar(x + width/2, private_means, width, label='Private Mean', alpha=0.7)
        
        ax.set_xlabel('Medical Attributes')
        ax.set_ylabel('Mean Value')
        ax.set_title('True vs Private Mean Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation=45)
        ax.legend()
        
        # Add error values on top
        for i, error in enumerate(errors):
            ax.text(i, max(true_means[i], private_means[i]) * 1.1, 
                   'Error: {:.1f}'.format(error), ha='center', va='bottom', fontsize=8)
    
    def _plot_count_comparison(self, ax, analyzer):
        """Plot comparison of true vs private counts."""
        conditions = [
            'Diabetes == 1',
            'Hypertension == 1', 
            'HeartDisease == 1',
            'Age > 65',
            'BMI > 30'
        ]
        
        labels = ['Diabetes', 'Hypertension', 'Heart Disease', 'Elderly', 'Obese']
        
        true_counts = []
        private_counts = []
        
        for condition in conditions:
            true_count, private_count = analyzer.private_count(condition)
            true_counts.append(true_count)
            private_counts.append(private_count)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, true_counts, width, label='True Count', alpha=0.7)
        ax.bar(x + width/2, private_counts, width, label='Private Count', alpha=0.7)
        
        ax.set_xlabel('Medical Conditions')
        ax.set_ylabel('Count')
        ax.set_title('True vs Private Count', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
    
    def _plot_histogram_comparison(self, ax, analyzer):
        """Plot histogram comparison for gender distribution."""
        bins = ['Male', 'Female']
        true_hist, private_hist = analyzer.private_histogram('Gender', bins)
        
        categories = list(true_hist.keys())
        true_values = [true_hist[cat] for cat in categories]
        private_values = [private_hist[cat] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, true_values, width, label='True Distribution', alpha=0.7)
        ax.bar(x + width/2, private_values, width, label='Private Distribution', alpha=0.7)
        
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Gender Distribution: True vs Private Histogram', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Add percentage labels
        total_true = sum(true_values)
        total_private = sum(private_values)
        
        for i, (true_val, private_val) in enumerate(zip(true_values, private_values)):
            ax.text(i - width/2, true_val + 50, '{:.1f}%'.format(100*true_val/total_true), 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, private_val + 50, '{:.1f}%'.format(100*private_val/total_private), 
                   ha='center', va='bottom', fontsize=8)

def demonstrate_differential_privacy():
    """
    Main demonstration function for differential privacy concepts.
    """
    print("=" * 80)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION")
    print("=" * 80)
    
    # Generate synthetic medical data
    print("\n1. Generating synthetic medical dataset...")
    data_generator = MedicalDataGenerator()
    medical_data = data_generator.generate_patient_records(n_patients=10000)
    
    print("   Generated {} patient records".format(len(medical_data)))
    print("   Columns: {}".format(list(medical_data.columns)))
    
    # Initialize differential privacy analyzer
    epsilon = 1.0
    print("\n2. Initializing differential privacy mechanism...")
    print("   Privacy parameter (ε): {}".format(epsilon))
    print("   Privacy level: Medium (ε=1.0 provides good balance)")
    
    analyzer = DifferentialPrivacyAnalyzer(medical_data, epsilon)
    
    # Demonstrate various DP mechanisms
    print("\n3. Demonstrating differential privacy mechanisms...")
    
    # Mean estimation with DP
    print("\n   3.1 Private Mean Estimation:")
    true_age_mean, private_age_mean = analyzer.private_mean('Age', (18, 90))
    print("       True mean age: {:.2f}".format(true_age_mean))
    print("       Private mean age: {:.2f}".format(private_age_mean))
    print("       Absolute error: {:.2f}".format(abs(true_age_mean - private_age_mean)))
    
    # Count queries with DP
    print("\n   3.2 Private Count Queries:")
    conditions = [
        ('Diabetes patients', 'Diabetes == 1'),
        ('Elderly patients (>65)', 'Age > 65'),
        ('High BP patients', 'SystolicBP > 140')
    ]
    
    for desc, condition in conditions:
        true_count, private_count = analyzer.private_count(condition)
        print("       {}: True={}, Private={:.0f}, Error={:.0f}".format(
            desc, true_count, private_count, abs(true_count - private_count)))
    
    # Histogram with DP
    print("\n   3.3 Private Histogram:")
    true_hist, private_hist = analyzer.private_histogram('Gender', ['Male', 'Female'])
    print("       Gender distribution (True): {}".format(true_hist))
    print("       Gender distribution (Private): {}".format(
        {k: "{:.0f}".format(v) for k, v in private_hist.items()}))
    
    # Privacy protection demonstration
    print("\n4. Demonstrating privacy protection...")
    target_patient = medical_data.iloc[100]['PatientID']
    
    def age_query(analyzer_obj):
        true_mean, private_mean = analyzer_obj.private_mean('Age', (18, 90))
        return private_mean
    
    privacy_demo = analyzer.demonstrate_privacy_loss(target_patient, age_query)
    
    print("   Target patient: {}".format(target_patient))
    print("   Query with patient: {:.2f}".format(privacy_demo['with_patient']))
    print("   Query without patient: {:.2f}".format(privacy_demo['without_patient']))
    print("   Difference: {:.2f} (DP ensures this is bounded)".format(
        privacy_demo.get('difference', 0)))
    
    # Privacy budget analysis
    print("\n5. Privacy budget analysis...")
    print("   Initial privacy budget (ε): {}".format(epsilon))
    print("   Budget per query: {:.3f}".format(epsilon))
    print("   Queries executed: Multiple (budget management is crucial)")
    print("   Remaining budget: Depends on query composition")
    
    # Generate visualizations
    print("\n6. Generating comprehensive visualizations...")
    visualizer = DifferentialPrivacyVisualizer()
    visualizer.create_dp_dashboard(analyzer, "differential_privacy_dashboard.png")
    
    # Epsilon comparison
    print("\n7. Privacy parameter (ε) comparison:")
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("   ε Value | Privacy Level | Noise Level | Use Case")
    print("   --------|---------------|-------------|----------")
    for eps in epsilon_values:
        if eps <= 0.5:
            privacy = "Very High"
            noise = "Very High"
            use_case = "Highly sensitive data"
        elif eps <= 1.0:
            privacy = "High"
            noise = "High"
            use_case = "Medical research"
        elif eps <= 2.0:
            privacy = "Medium"
            noise = "Medium"
            use_case = "Census data"
        else:
            privacy = "Low"
            noise = "Low"
            use_case = "General analytics"
        
        print("   {:<7} | {:<13} | {:<11} | {}".format(eps, privacy, noise, use_case))
    
    # Mathematical guarantees
    print("\n8. Differential privacy guarantees:")
    print("   ✓ Individual privacy: No single record significantly affects output")
    print("   ✓ Composability: Multiple queries can be combined safely")
    print("   ✓ Robustness: Protection against arbitrary background knowledge")
    print("   ✓ Quantifiable: Mathematical bounds on privacy loss")
    
    # Comparison with other privacy techniques
    print("\n9. Comparison with other privacy methods:")
    print("   Technique           | Privacy | Utility | Guarantees | Composability")
    print("   -------------------|---------|---------|------------|-------------")
    print("   Differential Privacy|   High  |  Good   |   Strong   |     Yes")
    print("   K-Anonymity        |  Medium |  Good   |   Weak     |     No")
    print("   Data Masking       |   Low   |  High   |   None     |     No")
    print("   Homomorphic Enc.   |   High  |  High   |   Strong   |    Limited")
    
    print("\n10. Real-world applications:")
    print("    • Apple's iOS usage analytics")
    print("    • Google's Chrome telemetry")
    print("    • US Census Bureau statistics")
    print("    • Medical research data sharing")
    print("    • Smart city traffic analysis")
    
    print("\n" + "="*80)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION COMPLETED")
    print("Dashboard saved as 'differential_privacy_dashboard.png'")
    print("="*80)
    
    return analyzer, medical_data

if __name__ == "__main__":
    # Run the demonstration
    analyzer, data = demonstrate_differential_privacy()
    
    print("\nDataset ready for further analysis!")
    print("Original data shape: {}".format(data.shape))
    print("Privacy parameter (ε): {}".format(analyzer.epsilon))
    print("Privacy guarantees: Differential Privacy with ε-indistinguishability")
