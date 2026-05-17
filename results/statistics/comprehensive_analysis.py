#!/usr/bin/env python3
"""
Comprehensive Statistical and TOPSIS Analysis for SpheroSeg Models
Combines all statistical tests and TOPSIS analysis into a unified pipeline.

Usage:
    python results/statistics/comprehensive_analysis.py [--data PATH]
"""

import argparse
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare, levene
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

# Default location of the merged per-image results bundled with this repo.
DEFAULT_DATA = (Path(__file__).resolve().parents[1]
                / "evaluation_results_20250829_120800"
                / "Merged_DTS_SpheroSeg_results"
                / "Merged_DTS_SpheroSeg_detailed_results.csv")


class ComprehensiveAnalysis:
    """Unified analysis combining statistical tests and TOPSIS evaluation"""

    def __init__(self, data_path=None):
        """Initialize with data loading and preprocessing."""
        if data_path is None:
            data_path = DEFAULT_DATA
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        self.setup_model_info()
        
    def prepare_data(self):
        """Prepare and clean the data"""
        # Extract model names and training types
        self.df['Model_Base'] = self.df['Model'].str.replace('_pretrained', '').str.replace('_finetuned', '')
        self.df['Training_Type'] = self.df['Model'].apply(
            lambda x: 'pretrained' if 'pretrained' in x else 'finetuned'
        )
        
        # Define model architectures
        self.architectures = [
            'cbam_unet', 'resunet_advanced', 'resunet_advanced_mini',
            'resunet_small', 'unet', 'hrnet', 'pspnet_new', 'lightm_unet'
        ]
        
        # Get aggregate statistics
        self.model_stats = self.df.groupby('Model').agg({
            'IoU': ['mean', 'std', 'median'],
            'Dice': ['mean', 'std'],
            'Inference_Time_ms': ['mean', 'std']
        }).round(4)
        
    def setup_model_info(self):
        """Setup model display names and colors"""
        self.model_display_names = {
            'cbam_unet_pretrained': 'CBAM-ResUNet',
            'resunet_advanced_pretrained': 'MA-ResUNet',
            'resunet_advanced_mini_pretrained': 'MA-ResUNet-Mini',
            'resunet_small_pretrained': 'LC-ResUNet',
            'unet_pretrained': 'U-Net',
            'hrnet_pretrained': 'HRNet',
            'pspnet_new_pretrained': 'PSPNet',
            'lightm_unet_pretrained': 'LightM-UNet'
        }
        
        self.model_colors = {
            'cbam_unet': '#E74C3C',
            'resunet_advanced': '#3498DB',
            'resunet_advanced_mini': '#9B59B6',
            'resunet_small': '#2ECC71',
            'unet': '#F39C12',
            'hrnet': '#1ABC9C',
            'pspnet_new': '#34495E',
            'lightm_unet': '#E67E22'
        }
    
    def run_all_statistical_tests(self):
        """Run all statistical tests and return results DataFrame"""
        results = []
        
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        # 1. Wilcoxon Signed-Rank Test (Paired comparison)
        print("\n1. WILCOXON SIGNED-RANK TEST (Paired Comparison)")
        print("-"*50)
        for arch in self.architectures:
            pretrained_model = f"{arch}_pretrained"
            finetuned_model = f"{arch}_finetuned"
            
            if pretrained_model in self.df['Model'].values and finetuned_model in self.df['Model'].values:
                pretrained_iou = self.df[self.df['Model'] == pretrained_model]['IoU'].values
                finetuned_iou = self.df[self.df['Model'] == finetuned_model]['IoU'].values
                
                if len(pretrained_iou) == len(finetuned_iou):
                    statistic, p_value = wilcoxon(pretrained_iou, finetuned_iou)
                    mean_diff = np.mean(pretrained_iou - finetuned_iou)
                    
                    results.append({
                        'Test': 'Wilcoxon Signed-Rank',
                        'Comparison': f'{arch} (pre vs fine)',
                        'Statistic': statistic,
                        'P-value': p_value,
                        'Mean_Diff': mean_diff,
                        'Significant': 'Yes' if p_value < 0.05 else 'No',
                        'Effect': 'Pretrained better' if mean_diff > 0 else 'Finetuned better'
                    })
                    
                    print(f"{arch:20} W={statistic:8.2f}, p={p_value:.4f}, "
                          f"Δ={mean_diff:+.4f} {'*' if p_value < 0.05 else ''}")
        
        # 2. Mann-Whitney U Test (Independent samples)
        print("\n2. MANN-WHITNEY U TEST (Independent Samples)")
        print("-"*50)
        pretrained_all = self.df[self.df['Training_Type'] == 'pretrained']['IoU'].values
        finetuned_all = self.df[self.df['Training_Type'] == 'finetuned']['IoU'].values
        
        u_stat, p_value = mannwhitneyu(pretrained_all, finetuned_all)
        
        # Calculate Cliff's Delta for effect size
        n1, n2 = len(pretrained_all), len(finetuned_all)
        cliff_delta = (2 * u_stat) / (n1 * n2) - 1
        
        results.append({
            'Test': 'Mann-Whitney U',
            'Comparison': 'All pretrained vs finetuned',
            'Statistic': u_stat,
            'P-value': p_value,
            'Mean_Diff': np.mean(pretrained_all) - np.mean(finetuned_all),
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Effect': f"Cliff's δ = {cliff_delta:.4f}"
        })
        
        print(f"U-statistic: {u_stat:.2f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Cliff's Delta: {cliff_delta:.4f}")
        print(f"Mean IoU - Pretrained: {np.mean(pretrained_all):.4f}, Finetuned: {np.mean(finetuned_all):.4f}")
        
        # 3. Levene's Test for Variance
        print("\n3. LEVENE'S TEST (Variance Comparison)")
        print("-"*50)
        levene_stat, levene_p = levene(pretrained_all, finetuned_all)
        
        results.append({
            'Test': "Levene's Test",
            'Comparison': 'Variance: pre vs fine',
            'Statistic': levene_stat,
            'P-value': levene_p,
            'Mean_Diff': np.var(pretrained_all) - np.var(finetuned_all),
            'Significant': 'Yes' if levene_p < 0.05 else 'No',
            'Effect': 'Unequal variances' if levene_p < 0.05 else 'Equal variances'
        })
        
        print(f"F-statistic: {levene_stat:.4f}")
        print(f"P-value: {levene_p:.6f}")
        print(f"Variance - Pretrained: {np.var(pretrained_all):.6f}, Finetuned: {np.var(finetuned_all):.6f}")
        
        # 4. Friedman Test (Multiple related samples)
        print("\n4. FRIEDMAN TEST (Multiple Related Samples)")
        print("-"*50)
        # Prepare data for Friedman test - use ALL samples
        model_iou_matrix = []
        min_samples = min([len(self.df[self.df['Model'] == m]['IoU']) for m in self.df['Model'].unique()])
        print(f"Using all {min_samples} samples for Friedman test")

        for model in self.df['Model'].unique():
            model_iou_matrix.append(self.df[self.df['Model'] == model]['IoU'].values[:min_samples])  # Use all available samples
        
        if len(model_iou_matrix) > 2:
            friedman_stat, friedman_p = friedmanchisquare(*model_iou_matrix)
            
            results.append({
                'Test': 'Friedman Test',
                'Comparison': 'All 16 models',
                'Statistic': friedman_stat,
                'P-value': friedman_p,
                'Mean_Diff': np.nan,
                'Significant': 'Yes' if friedman_p < 0.05 else 'No',
                'Effect': 'Models differ' if friedman_p < 0.05 else 'Models similar'
            })
            
            print(f"Chi-square statistic: {friedman_stat:.2f}")
            print(f"P-value: {friedman_p:.2e}")
        
        # 5. Additional Friedman Test for Architectures only
        print("\n5. FRIEDMAN TEST FOR ARCHITECTURES (Pretrained Only)")
        print("-"*50)
        # Compare only pretrained versions to avoid duplication
        pretrained_models = [f"{arch}_pretrained" for arch in self.architectures 
                           if f"{arch}_pretrained" in self.df['Model'].unique()]
        
        arch_iou_matrix = []
        min_samples = min([len(self.df[self.df['Model'] == m]['IoU']) for m in pretrained_models])
        
        for model in pretrained_models:
            arch_iou_matrix.append(self.df[self.df['Model'] == model]['IoU'].values[:min_samples])
        
        if len(arch_iou_matrix) > 2:
            arch_friedman_stat, arch_friedman_p = friedmanchisquare(*arch_iou_matrix)
            
            results.append({
                'Test': 'Friedman (Architectures)',
                'Comparison': '8 pretrained architectures',
                'Statistic': arch_friedman_stat,
                'P-value': arch_friedman_p,
                'Mean_Diff': np.nan,
                'Significant': 'Yes' if arch_friedman_p < 0.05 else 'No',
                'Effect': 'Architectures differ' if arch_friedman_p < 0.05 else 'Similar performance'
            })
            
            print(f"Chi-square statistic: {arch_friedman_stat:.2f}")
            print(f"P-value: {arch_friedman_p:.2e}")
            print("Note: Using Friedman test (correct for repeated measures)")
        
        return pd.DataFrame(results)
    
    def calculate_topsis(self, accuracy_weight=0.6):
        """Calculate TOPSIS scores for model ranking"""
        # Get mean statistics per model
        topsis_data = []
        for model in self.df['Model'].unique():
            model_data = self.df[self.df['Model'] == model]
            topsis_data.append({
                'Model': model,
                'IoU': model_data['IoU'].mean(),
                'Inference_Time_ms': model_data['Inference_Time_ms'].mean()
            })
        
        model_stats = pd.DataFrame(topsis_data)
        speed_weight = 1 - accuracy_weight
        
        # Step 1: Vector normalization
        iou_sum_squares = np.sqrt((model_stats['IoU']**2).sum())
        model_stats['IoU_norm'] = model_stats['IoU'] / iou_sum_squares
        
        time_sum_squares = np.sqrt((model_stats['Inference_Time_ms']**2).sum())
        model_stats['Time_norm'] = model_stats['Inference_Time_ms'] / time_sum_squares
        
        # Step 2: Apply weights
        model_stats['IoU_weighted'] = model_stats['IoU_norm'] * accuracy_weight
        model_stats['Time_weighted'] = model_stats['Time_norm'] * speed_weight
        
        # Step 3: Determine ideal best and worst
        ideal_best = {
            'IoU': model_stats['IoU_weighted'].max(),
            'Time': model_stats['Time_weighted'].min()
        }
        ideal_worst = {
            'IoU': model_stats['IoU_weighted'].min(),
            'Time': model_stats['Time_weighted'].max()
        }
        
        # Step 4: Calculate distances
        model_stats['S_plus'] = np.sqrt(
            (model_stats['IoU_weighted'] - ideal_best['IoU'])**2 + 
            (model_stats['Time_weighted'] - ideal_best['Time'])**2
        )
        model_stats['S_minus'] = np.sqrt(
            (model_stats['IoU_weighted'] - ideal_worst['IoU'])**2 + 
            (model_stats['Time_weighted'] - ideal_worst['Time'])**2
        )
        
        # Step 5: Calculate TOPSIS score
        model_stats['TOPSIS_Score'] = model_stats['S_minus'] / (
            model_stats['S_plus'] + model_stats['S_minus'] + 1e-10
        )
        
        # Sort by TOPSIS score
        model_stats = model_stats.sort_values('TOPSIS_Score', ascending=False)
        model_stats['Rank'] = range(1, len(model_stats) + 1)
        
        return model_stats
    
    def run_topsis_analysis(self):
        """Run TOPSIS analysis with multiple weight scenarios"""
        print("\n" + "="*80)
        print("TOPSIS MULTI-CRITERIA DECISION ANALYSIS")
        print("="*80)
        
        scenarios = [
            ("Balanced (60% acc, 40% speed)", 0.6),
            ("Accuracy-focused (80% acc, 20% speed)", 0.8),
            ("Speed-focused (40% acc, 60% speed)", 0.4),
            ("Clinical use (70% acc, 30% speed)", 0.7)
        ]
        
        topsis_results = []
        
        for scenario_name, weight in scenarios:
            print(f"\n{scenario_name}")
            print("-"*50)
            
            scenario_stats = self.calculate_topsis(accuracy_weight=weight)
            top_3 = scenario_stats.head(3)
            
            for idx, row in top_3.iterrows():
                display_name = self.model_display_names.get(row['Model'], row['Model'])
                if 'pretrained' not in row['Model']:
                    display_name = row['Model'].replace('_finetuned', '') + ' (FT)'
                
                print(f"  {row['Rank']}. {display_name:20} Score: {row['TOPSIS_Score']:.4f} "
                      f"(IoU: {row['IoU']:.4f}, Time: {row['Inference_Time_ms']:.1f}ms)")
                
                topsis_results.append({
                    'Scenario': scenario_name,
                    'Rank': row['Rank'],
                    'Model': display_name,
                    'TOPSIS_Score': row['TOPSIS_Score'],
                    'IoU': row['IoU'],
                    'Time_ms': row['Inference_Time_ms']
                })
        
        return pd.DataFrame(topsis_results)
    
    def create_performance_plot(self, save_path='performance_efficiency_plot.png'):
        """Create the performance vs efficiency visualization"""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Calculate TOPSIS for plot
        model_stats = self.calculate_topsis(accuracy_weight=0.6)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Separate pretrained and finetuned
        pretrained = model_stats[model_stats['Model'].str.contains('pretrained')]
        
        # Plot all models
        for _, row in model_stats.iterrows():
            model_base = row['Model'].replace('_pretrained', '').replace('_finetuned', '')
            color = self.model_colors.get(model_base, '#95A5A6')
            
            if 'pretrained' in row['Model']:
                marker = 'o'
                size = 150
                alpha = 0.8
                edgecolor = 'black'
                linewidth = 2
            else:
                marker = 'o'
                size = 150
                alpha = 0.4
                edgecolor = 'gray'
                linewidth = 1
            
            ax.scatter(row['Inference_Time_ms'], row['IoU'], 
                      c=[color], s=size, marker=marker, alpha=alpha,
                      edgecolors=edgecolor, linewidth=linewidth)
        
        # Annotate pretrained models
        for _, row in pretrained.iterrows():
            display_name = self.model_display_names.get(row['Model'], row['Model'])
            ax.annotate(display_name,
                       (row['Inference_Time_ms'], row['IoU']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Styling
        ax.set_xlabel('Inference Time (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('IoU Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance-Efficiency Analysis for Spheroid Segmentation Models',
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(model_stats['Inference_Time_ms']) * 1.1)
        ax.set_ylim(0.89, 0.94)
        
        # Add regions
        ax.axhspan(0.92, 0.94, alpha=0.1, color='green', label='High Accuracy Region')
        ax.axvspan(0, 50, alpha=0.1, color='blue', label='Real-time Region (<50ms)')
        
        # Legend
        legend_elements = [
            mpatches.Circle((0, 0), 1, facecolor='gray', edgecolor='black', 
                          linewidth=2, alpha=0.8, label='Pretrained'),
            mpatches.Circle((0, 0), 1, facecolor='gray', edgecolor='gray', 
                          linewidth=1, alpha=0.4, label='Fine-tuned')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
        
        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig
    
    def save_all_results(self):
        """Save all test results to unified CSV file"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Run all analyses
        statistical_tests = self.run_all_statistical_tests()
        topsis_results = self.run_topsis_analysis()
        
        # Save statistical tests
        statistical_tests.to_csv('tests.csv', index=False)
        print("\n✓ Statistical tests saved to: tests.csv")
        
        # Save TOPSIS results
        topsis_results.to_csv('topsis_rankings.csv', index=False)
        print("✓ TOPSIS rankings saved to: topsis_rankings.csv")
        
        # Save full TOPSIS scores
        full_topsis = self.calculate_topsis(accuracy_weight=0.6)
        full_topsis.to_csv('topsis_full_scores.csv', index=False)
        print("✓ Full TOPSIS scores saved to: topsis_full_scores.csv")
        
        # Create visualization
        self.create_performance_plot()
        print("✓ Performance plot saved to: performance_efficiency_plot.png")
        
        # Create summary statistics
        summary = self.model_stats.round(4)
        summary.to_csv('model_summary_statistics.csv')
        print("✓ Model summary statistics saved to: model_summary_statistics.csv")
        
        return statistical_tests, topsis_results
    
    def calculate_bootstrap_ci(self, data, n_resamples=10000):
        """Calculate bootstrap confidence interval for a dataset"""
        from scipy import stats
        ci = stats.bootstrap((data,), np.mean,
                            n_resamples=n_resamples,
                            confidence_level=0.95,
                            method='percentile')
        return ci.confidence_interval.low, ci.confidence_interval.high

    def generate_manuscript_tables(self):
        """Generate all tables needed for the manuscript"""
        print("\n" + "="*80)
        print("MANUSCRIPT TABLES GENERATION")
        print("="*80)

        # Table 4: Model Performance
        print("\n### TABLE 4: MODEL PERFORMANCE (Full Test Set, n=1,019) ###")
        print("-"*80)
        print(f"{'Model':<20} {'Training':<12} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Time(ms)':<8}")
        print("-"*80)
        
        models_order = [
            ('cbam_unet_pretrained', 'cbam_unet_finetuned'),
            ('unet_pretrained', 'unet_finetuned'),
            ('resunet_advanced_pretrained', 'resunet_advanced_finetuned'),
            ('resunet_advanced_mini_pretrained', 'resunet_advanced_mini_finetuned'),
            ('hrnet_pretrained', 'hrnet_finetuned'),
            ('resunet_small_pretrained', 'resunet_small_finetuned'),
            ('pspnet_new_pretrained', 'pspnet_new_finetuned'),
            ('lightm_unet_pretrained', 'lightm_unet_finetuned')
        ]
        
        table4_data = []
        for pretrained_model, finetuned_model in models_order:
            for model in [pretrained_model, finetuned_model]:
                if model in self.df['Model'].values:
                    model_data = self.df[self.df['Model'] == model]

                    # Calculate bootstrap CI for IoU (10,000 iterations)
                    iou_ci_low, iou_ci_high = self.calculate_bootstrap_ci(
                        model_data['IoU'].values,
                        n_resamples=10000
                    )

                    row = {
                        'Model': self.model_display_names.get(model, model.replace('_pretrained', '').replace('_finetuned', '')),
                        'Training': 'Pretrained' if 'pretrained' in model else 'Fine-tuned',
                        'IoU': model_data['IoU'].mean(),
                        'IoU_CI_low': iou_ci_low,
                        'IoU_CI_high': iou_ci_high,
                        'Dice': model_data['Dice'].mean(),
                        'Precision': model_data['Precision'].mean(),
                        'Recall': model_data['Recall'].mean(),
                        'Time_ms': model_data['Inference_Time_ms'].mean()
                    }
                    table4_data.append(row)
                    
                    print(f"{row['Model']:<20} {row['Training']:<12} {row['IoU']:.4f}   {row['Dice']:.4f}   "
                          f"{row['Precision']:.4f}     {row['Recall']:.4f}   {row['Time_ms']:.1f}")
        
        # Save Table 4
        pd.DataFrame(table4_data).to_csv('table4_model_performance.csv', index=False)

        # Print bootstrap CI for key models (as mentioned in manuscript)
        print("\n### BOOTSTRAP CONFIDENCE INTERVALS (10,000 iterations) ###")
        for row in table4_data:
            if row['Model'] == 'CBAM-ResUNet' and row['Training'] == 'Pretrained':
                print(f"CBAM-ResUNet (pretrained): IoU = {row['IoU']:.4f}, 95% CI: [{row['IoU_CI_low']:.4f}-{row['IoU_CI_high']:.4f}]")
                # Check if it matches manuscript claim [0.9292-0.9391]
                if abs(row['IoU_CI_low'] - 0.9292) < 0.01 and abs(row['IoU_CI_high'] - 0.9391) < 0.01:
                    print("✅ Matches manuscript claim [0.9292-0.9391]")
                else:
                    print(f"⚠️ Manuscript claims [0.9292-0.9391], calculated [{row['IoU_CI_low']:.4f}-{row['IoU_CI_high']:.4f}]")
        
        # Table 7: Cross-dataset Generalization
        print("\n### TABLE 7: CROSS-DATASET GENERALIZATION ###")
        print("-"*80)
        print(f"{'Model':<25} {'BxPC-3 IoU':<20} {'DTS IoU':<20} {'Δ IoU':<10}")
        print("-"*80)
        
        table7_data = []
        pretrained_models = ['cbam_unet_pretrained', 'resunet_advanced_pretrained', 
                           'resunet_advanced_mini_pretrained', 'resunet_small_pretrained',
                           'unet_pretrained', 'hrnet_pretrained', 'pspnet_new_pretrained',
                           'lightm_unet_pretrained']
        
        for model in pretrained_models:
            if model in self.df['Model'].values:
                # DTS dataset has numeric filenames (1.png to 366.png)
                # BxPC-3 has filenames containing 'bxpc-3'
                is_dts = self.df['Filename'].apply(lambda x: x.replace('.png', '').replace('.jpg', '').isdigit())
                is_bxpc3 = ~is_dts  # Everything that's not DTS is BxPC-3
                
                bxpc3_data = self.df[(self.df['Model'] == model) & is_bxpc3]
                dts_data = self.df[(self.df['Model'] == model) & is_dts]
                
                if len(bxpc3_data) > 0 and len(dts_data) > 0:
                    bxpc3_iou = bxpc3_data['IoU'].mean()
                    dts_iou = dts_data['IoU'].mean()
                    
                    # Calculate 95% CI using bootstrap (10,000 iterations as per manuscript)
                    from scipy import stats
                    bxpc3_ci = stats.bootstrap((bxpc3_data['IoU'].values,), np.mean,
                                              n_resamples=10000, confidence_level=0.95, method='percentile')
                    dts_ci = stats.bootstrap((dts_data['IoU'].values,), np.mean,
                                            n_resamples=10000, confidence_level=0.95, method='percentile')
                    
                    display_name = self.model_display_names.get(model, model)
                    
                    row = {
                        'Model': display_name,
                        'BxPC3_IoU': bxpc3_iou,
                        'BxPC3_CI_lower': bxpc3_ci.confidence_interval.low,
                        'BxPC3_CI_upper': bxpc3_ci.confidence_interval.high,
                        'DTS_IoU': dts_iou,
                        'DTS_CI_lower': dts_ci.confidence_interval.low,
                        'DTS_CI_upper': dts_ci.confidence_interval.high,
                        'Delta_IoU': bxpc3_iou - dts_iou
                    }
                    table7_data.append(row)
                    
                    print(f"{display_name:<25} {bxpc3_iou:.3f} ({bxpc3_ci.confidence_interval.low:.3f}-{bxpc3_ci.confidence_interval.high:.3f})   "
                          f"{dts_iou:.3f} ({dts_ci.confidence_interval.low:.3f}-{dts_ci.confidence_interval.high:.3f})   "
                          f"{bxpc3_iou - dts_iou:+.3f}")
        
        # Save Table 7
        pd.DataFrame(table7_data).to_csv('table7_generalization.csv', index=False)
        
        # Additional statistics for manuscript text
        print("\n### KEY STATISTICS FOR MANUSCRIPT TEXT ###")
        print("-"*80)
        
        # Overall statistics
        pretrained_all = self.df[self.df['Model'].str.contains('pretrained')]['IoU'].values
        finetuned_all = self.df[~self.df['Model'].str.contains('pretrained')]['IoU'].values
        
        print(f"Pretrained mean IoU: {np.mean(pretrained_all):.4f} ± {np.std(pretrained_all):.4f}")
        print(f"Fine-tuned mean IoU: {np.mean(finetuned_all):.4f} ± {np.std(finetuned_all):.4f}")
        print(f"Difference: {np.mean(pretrained_all) - np.mean(finetuned_all):+.4f}")
        
        # Best models
        best_iou_model = self.df.groupby('Model')['IoU'].mean().idxmax()
        best_iou_value = self.df.groupby('Model')['IoU'].mean().max()
        print(f"\nBest IoU: {self.model_display_names.get(best_iou_model, best_iou_model)} = {best_iou_value:.4f}")
        
        fastest_model = self.df.groupby('Model')['Inference_Time_ms'].mean().idxmin()
        fastest_time = self.df.groupby('Model')['Inference_Time_ms'].mean().min()
        print(f"Fastest: {self.model_display_names.get(fastest_model, fastest_model)} = {fastest_time:.1f}ms")
        
        # Failure analysis (IoU < 0.7)
        print("\n### FAILURE ANALYSIS (IoU < 0.7) ###")
        failure_data = []
        all_models = self.df['Model'].unique()
        
        for model in all_models:
            if model in self.df['Model'].values:
                failures = len(self.df[(self.df['Model'] == model) & (self.df['IoU'] < 0.7)])
                total = len(self.df[self.df['Model'] == model])
                failure_rate = (failures / total) * 100
                
                failure_data.append({
                    'Model': self.model_display_names.get(model, model.replace('_pretrained', '').replace('_finetuned', '')),
                    'Training': 'Pretrained' if 'pretrained' in model else 'Fine-tuned',
                    'Failures': failures,
                    'Total': total,
                    'Failure_Rate_%': failure_rate
                })
                
                if model in ['cbam_unet_pretrained', 'lightm_unet_pretrained', 'hrnet_pretrained', 'unet_pretrained']:
                    print(f"{self.model_display_names.get(model, model)}: {failure_rate:.1f}% ({failures}/{total})")
        
        # Save failure analysis
        failure_df = pd.DataFrame(failure_data).sort_values('Failure_Rate_%')
        failure_df.to_csv('failure_analysis.csv', index=False)
        print(f"\nLowest failure rate: {failure_df.iloc[0]['Model']} ({failure_df.iloc[0]['Training']}) = {failure_df.iloc[0]['Failure_Rate_%']:.1f}%")
        print(f"Highest failure rate: {failure_df.iloc[-1]['Model']} ({failure_df.iloc[-1]['Training']}) = {failure_df.iloc[-1]['Failure_Rate_%']:.1f}%")
        
        # Performance-efficiency ratios
        print("\n### PERFORMANCE-EFFICIENCY RATIOS ###")
        for model in ['hrnet_pretrained', 'unet_pretrained', 'cbam_unet_pretrained']:
            if model in self.df['Model'].values:
                model_data = self.df[self.df['Model'] == model]
                iou = model_data['IoU'].mean()
                time = model_data['Inference_Time_ms'].mean()
                ratio = (iou / time) * 1000
                fps = 1000 / time
                print(f"{self.model_display_names.get(model, model)}: Ratio={ratio:.2f}, FPS={fps:.1f}")
        
        return table4_data, table7_data
    
    def print_key_findings(self):
        """Print key findings from the analysis"""
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        # Best models by different criteria
        topsis_balanced = self.calculate_topsis(accuracy_weight=0.6)
        best_balanced = topsis_balanced.iloc[0]
        
        print("\n1. BEST OVERALL MODEL (TOPSIS 60/40):")
        print(f"   {self.model_display_names.get(best_balanced['Model'], best_balanced['Model'])}")
        print(f"   - TOPSIS Score: {best_balanced['TOPSIS_Score']:.4f}")
        print(f"   - IoU: {best_balanced['IoU']:.4f}")
        print(f"   - Inference Time: {best_balanced['Inference_Time_ms']:.1f}ms")
        
        # Best accuracy
        best_accuracy = self.model_stats['IoU']['mean'].idxmax()
        print(f"\n2. HIGHEST ACCURACY MODEL:")
        print(f"   {self.model_display_names.get(best_accuracy, best_accuracy)}")
        print(f"   - IoU: {self.model_stats.loc[best_accuracy, ('IoU', 'mean')]:.4f}")
        
        # Fastest model
        fastest = self.model_stats['Inference_Time_ms']['mean'].idxmin()
        print(f"\n3. FASTEST MODEL:")
        print(f"   {self.model_display_names.get(fastest, fastest)}")
        print(f"   - Inference Time: {self.model_stats.loc[fastest, ('Inference_Time_ms', 'mean')]:.1f}ms")
        
        # Pretrained vs Finetuned
        pretrained_mean = self.df[self.df['Training_Type'] == 'pretrained']['IoU'].mean()
        finetuned_mean = self.df[self.df['Training_Type'] == 'finetuned']['IoU'].mean()
        
        print(f"\n4. TRAINING STRATEGY COMPARISON:")
        print(f"   Pretrained Mean IoU: {pretrained_mean:.4f}")
        print(f"   Finetuned Mean IoU: {finetuned_mean:.4f}")
        print(f"   Difference: {pretrained_mean - finetuned_mean:+.4f}")
        print(f"   Winner: {'Pretrained' if pretrained_mean > finetuned_mean else 'Finetuned'}")


def main():
    """Main execution function"""
    print("╔" + "═"*78 + "╗")
    print("║" + " COMPREHENSIVE ANALYSIS FOR SPHEROSEG MODELS ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    ap = argparse.ArgumentParser(description="SpheroSeg comprehensive analysis")
    ap.add_argument("--data", type=Path, default=None,
                    help=f"CSV with per-image IoU/Dice/Inference_Time_ms (default: {DEFAULT_DATA})")
    args, _ = ap.parse_known_args()
    analyzer = ComprehensiveAnalysis(args.data)
    
    # Run analyses and save results
    statistical_tests, topsis_results = analyzer.save_all_results()
    
    # Generate manuscript tables
    analyzer.generate_manuscript_tables()
    
    # Print key findings
    analyzer.print_key_findings()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • tests.csv - All statistical test results")
    print("  • topsis_rankings.csv - TOPSIS rankings for different scenarios")
    print("  • topsis_full_scores.csv - Complete TOPSIS scores")
    print("  • model_summary_statistics.csv - Aggregate model statistics")
    print("  • table4_model_performance.csv - Table 4 for manuscript")
    print("  • table7_generalization.csv - Table 7 for manuscript")
    print("  • failure_analysis.csv - Failure rate analysis (IoU < 0.7)")
    print("  • performance_efficiency_plot.png - Visualization")


if __name__ == "__main__":
    main()