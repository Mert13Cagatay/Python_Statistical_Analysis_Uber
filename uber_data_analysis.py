#!/usr/bin/env python3
"""
Python Uber Data Analysis - Comprehensive Statistical Analysis
Professional data quality assessment and outlier detection for Uber ride data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, kstest, jarque_bera, anderson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Grafik ayarları
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def download_dataset():
    """Download Uber dataset from Kaggle"""
    print("Downloading dataset...")
    try:
        path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")
        print(f"Dataset successfully downloaded: {path}")
        return path
    except Exception as e:
        print(f"Dataset download error: {e}")
        return None

def load_and_explore_data(dataset_path):
    """Load dataset and perform basic exploratory analysis"""
    if not dataset_path:
        print("Invalid dataset path!")
        return None
    
    # List files in dataset directory
    dataset_dir = Path(dataset_path)
    print(f"\nDataset directory: {dataset_dir}")
    print("Files in directory:")
    
    csv_files = []
    for file in dataset_dir.iterdir():
        if file.is_file():
            print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            if file.suffix.lower() == '.csv':
                csv_files.append(file)
    
    if not csv_files:
        print("No CSV file found!")
        return None
    
    # İlk CSV dosyasını yükle (genellikle ana veri dosyası)
    main_data_file = csv_files[0]
    print(f"\nAna veri dosyası yükleniyor: {main_data_file.name}")
    
    try:
        df = pd.read_csv(main_data_file)
        print(f"Veri başarıyla yüklendi! Boyut: {df.shape}")
        return df, csv_files
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None, []

def basic_data_analysis(df):
    """Temel veri analizi ve özet istatistikler"""
    print("\n" + "="*60)
    print("BASIC DATA ANALYSIS")
    print("="*60)
    
    # Veri seti hakkında genel bilgi
    print(f"\nVeri Seti Boyutu: {df.shape[0]:,} satır, {df.shape[1]} sütun")
    print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Sütun bilgileri
    print("\nColumn Information:")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        print(f"{col:20} | {str(dtype):12} | Null: {null_count:6} ({null_pct:5.1f}%) | Unique: {unique_count:6}")
    
    # İlk ve son birkaç satır
    print("\nFirst 5 Rows:")
    print("-" * 40)
    print(df.head())
    
    print("\nLast 5 Rows:")
    print("-" * 40)
    print(df.tail())
    
    # Sayısal sütunlar için özet istatistikler
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric Columns - Summary Statistics:")
        print("-" * 40)
        print(df[numeric_cols].describe())
    
    # Kategorik sütunlar için en sık değerler
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Columns - Most Frequent Values:")
        print("-" * 40)
        for col in categorical_cols[:5]:  # İlk 5 kategorik sütun
            print(f"\n{col}:")
            print(df[col].value_counts().head())
    
    return df

def create_initial_visualizations(df):
    """İlk görselleştirmeleri oluştur"""
    print("\n" + "="*60)
    print("BASIC VISUALIZATIONS")
    print("="*60)
    
    # Sayısal sütunlar için histogramlar
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Dağılımı')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frekans')
        
        # Kullanılmayan subplot'ları gizle
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('/Users/mertcagatay/Desktop/BitirmeDenemeler/numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Distribution plots for numeric variables created.")
    
    # Eksik değer analizi
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(12, 6))
        null_counts[null_counts > 0].plot(kind='bar')
        plt.title('Sütunlara Göre Eksik Değer Sayıları')
        plt.xlabel('Sütunlar')
        plt.ylabel('Eksik Değer Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/Users/mertcagatay/Desktop/BitirmeDenemeler/missing_values.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Missing value analysis plot created.")

def statistical_reliability_tests(df):
    """Veri güvenirliği için kapsamlı istatistiksel testler"""
    print("\n" + "="*80)
    print("STATISTICAL RELIABILITY TESTS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}
    
    if len(numeric_cols) == 0:
        print("❌ No numeric columns found!")
        return results
    
    # 1. NORMALLİK TESTLERİ
    print("\n🔍 1. NORMALITY TESTS")
    print("-" * 50)
    
    normality_results = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
            
        print(f"\n📊 {col} sütunu:")
        
        # Shapiro-Wilk Test (n < 5000 için ideal)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data.sample(min(5000, len(data))))
            print(f"   Shapiro-Wilk: stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            is_normal_shapiro = shapiro_p > 0.05
        else:
            is_normal_shapiro = None
            print(f"   Shapiro-Wilk: Veri çok büyük (n={len(data)})")
        
        # Kolmogorov-Smirnov Test
        ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
        print(f"   Kolmogorov-Smirnov: stat={ks_stat:.4f}, p-value={ks_p:.4f}")
        is_normal_ks = ks_p > 0.05
        
        # Jarque-Bera Test
        jb_stat, jb_p = jarque_bera(data)
        print(f"   Jarque-Bera: stat={jb_stat:.4f}, p-value={jb_p:.4f}")
        is_normal_jb = jb_p > 0.05
        
        # Anderson-Darling Test
        ad_result = anderson(data, dist='norm')
        is_normal_ad = ad_result.statistic < ad_result.critical_values[2]  # %5 seviyesi
        print(f"   Anderson-Darling: stat={ad_result.statistic:.4f}")
        
        # Özet
        normal_tests = [is_normal_ks, is_normal_jb, is_normal_ad]
        if is_normal_shapiro is not None:
            normal_tests.append(is_normal_shapiro)
        
        normal_ratio = sum(normal_tests) / len(normal_tests)
        
        if normal_ratio >= 0.5:
            print(f"   ✅ Sonuç: Normal dağılım ({normal_ratio:.1%} testler normal)")
        else:
            print(f"   ❌ Sonuç: Normal değil ({normal_ratio:.1%} testler normal)")
        
        normality_results[col] = {
            'shapiro_p': shapiro_p if is_normal_shapiro is not None else None,
            'ks_p': ks_p,
            'jb_p': jb_p,
            'anderson_stat': ad_result.statistic,
            'is_normal': normal_ratio >= 0.5
        }
    
    # 2. OUTLIER TESPİTİ
    print("\n🎯 2. OUTLIER DETECTION")
    print("-" * 50)
    
    outlier_results = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
            
        print(f"\n📊 {col} sütunu:")
        
        # IQR Yöntemi
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        iqr_percentage = (iqr_outliers / len(data)) * 100
        
        # Z-Score Yöntemi
        z_scores = np.abs(stats.zscore(data))
        z_outliers = (z_scores > 3).sum()
        z_percentage = (z_outliers / len(data)) * 100
        
        # Modified Z-Score (Robust)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
        modified_z_percentage = (modified_z_outliers / len(data)) * 100
        
        print(f"   IQR Yöntemi: {iqr_outliers} aykırı değer ({iqr_percentage:.2f}%)")
        print(f"   Z-Score: {z_outliers} aykırı değer ({z_percentage:.2f}%)")
        print(f"   Modified Z-Score: {modified_z_outliers} aykırı değer ({modified_z_percentage:.2f}%)")
        
        # Outlier değerlendirmesi
        avg_outlier_pct = (iqr_percentage + z_percentage + modified_z_percentage) / 3
        if avg_outlier_pct < 5:
            print(f"   ✅ Kabul edilebilir seviye ({avg_outlier_pct:.1f}% ortalama)")
        elif avg_outlier_pct < 10:
            print(f"   ⚠️  Dikkat gerekli ({avg_outlier_pct:.1f}% ortalama)")
        else:
            print(f"   ❌ Yüksek aykırı değer oranı ({avg_outlier_pct:.1f}% ortalama)")
        
        outlier_results[col] = {
            'iqr_count': iqr_outliers,
            'iqr_percentage': iqr_percentage,
            'z_count': z_outliers,
            'z_percentage': z_percentage,
            'modified_z_count': modified_z_outliers,
            'modified_z_percentage': modified_z_percentage,
            'average_percentage': avg_outlier_pct
        }
    
    # 3. KORELASYON VE ÇOKLU BAĞINTI ANALİZİ
    print("\n🔗 3. KORELASYON VE ÇOKLU BAĞINTI ANALİZİ")
    print("-" * 50)
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Yüksek korelasyonları bul
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_val
                ))
    
    print(f"Toplam {len(numeric_cols)} sayısal değişken arasında:")
    if high_corr_pairs:
        print("⚠️  Yüksek korelasyon tespit edildi (|r| > 0.8):")
        for var1, var2, corr in high_corr_pairs:
            print(f"   {var1} ↔ {var2}: r = {corr:.3f}")
    else:
        print("✅ Yüksek korelasyon (|r| > 0.8) tespit edilmedi")
    
    # VIF (Variance Inflation Factor) hesaplama
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Eksik değerleri çıkar ve geçerli sütunları filtrele
        clean_df = df[numeric_cols].dropna()
        
        if len(clean_df) == 0:
            print("⚠️  VIF hesaplaması için yeterli temiz veri yok")
        elif len(clean_df.columns) < 2:
            print("⚠️  VIF hesaplaması için en az 2 değişken gerekli")
        else:
            # Sabit olmayan sütunları bul
            non_constant_cols = []
            for col in clean_df.columns:
                if clean_df[col].nunique() > 1:  # Sabit olmayan sütunlar
                    non_constant_cols.append(col)
            
            if len(non_constant_cols) < 2:
                print("⚠️  VIF hesaplaması için en az 2 değişken sütun gerekli")
            else:
                # Standardize et
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clean_df[non_constant_cols])
                
                vif_data = pd.DataFrame()
                vif_data["Değişken"] = non_constant_cols
                vif_data["VIF"] = [variance_inflation_factor(scaled_data, i) 
                                  for i in range(len(non_constant_cols))]
                
                print(f"\nVIF (Variance Inflation Factor) Değerleri:")
                print(vif_data.to_string(index=False))
                
                high_vif = vif_data[vif_data["VIF"] > 10]
                if len(high_vif) > 0:
                    print("⚠️  Yüksek VIF değerleri (>10) - Çoklu bağıntı riski:")
                    print(high_vif.to_string(index=False))
                else:
                    print("✅ Tüm VIF değerleri kabul edilebilir seviyede (<10)")
            
    except ImportError:
        print("⚠️  VIF hesaplaması için statsmodels gerekli (pip install statsmodels)")
    except Exception as e:
        print(f"⚠️  VIF hesaplama hatası: {e}")
    
    # 4. VERİ TUTARLILIĞI TESTLERİ
    print("\n📋 4. DATA CONSISTENCY TESTS")
    print("-" * 50)
    
    consistency_issues = []
    
    # Negatif değerler kontrolü (olmaması gereken yerlerde)
    negative_checks = ['fare_amount', 'distance', 'duration', 'tip_amount']
    for col in negative_checks:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                consistency_issues.append(f"{col}: {negative_count} negatif değer")
    
    # Sıfır değerler kontrolü
    zero_checks = ['fare_amount', 'distance']
    for col in zero_checks:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            if zero_percentage > 5:
                consistency_issues.append(f"{col}: {zero_count} sıfır değer ({zero_percentage:.1f}%)")
    
    # Aşırı yüksek değerler
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            upper_99 = data.quantile(0.99)
            upper_999 = data.quantile(0.999)
            extreme_count = (data > upper_999 * 10).sum()  # %99.9'dan 10 kat büyük
            if extreme_count > 0:
                consistency_issues.append(f"{col}: {extreme_count} aşırı yüksek değer")
    
    if consistency_issues:
        print("⚠️  Tutarlılık sorunları tespit edildi:")
        for issue in consistency_issues:
            print(f"   - {issue}")
    else:
        print("✅ Veri tutarlılığı kontrollerinde sorun bulunamadı")
    
    # Sonuçları toplat
    results = {
        'normality': normality_results,
        'outliers': outlier_results,
        'high_correlations': high_corr_pairs,
        'consistency_issues': consistency_issues
    }
    
    return results

def clean_outliers(df, method='iqr', threshold=1.5):
    """Outlier'ları temizle ve temizlenmiş veri setini döndür"""
    print(f"\n🧹 OUTLIER TEMİZLEME - {method.upper()} Yöntemi")
    print("-" * 50)
    
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    removed_count = 0
    
    outlier_summary = {}
    
    for col in numeric_cols:
        if col not in df_clean.columns:
            continue
            
        original_count = len(df_clean)
        data = df_clean[col].dropna()
        
        if len(data) < 3:
            continue
            
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Outlier'ları tespit et - sadece null olmayan değerlerde
            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            # NaN değerleri False olarak ayarla
            outlier_mask = outlier_mask.fillna(False)
            
        elif method == 'zscore':
            # Z-score için sadece null olmayan değerleri kullan
            valid_indices = df_clean[col].notna()
            if valid_indices.sum() < 3:
                continue
                
            z_scores = pd.Series(index=df_clean.index, dtype=float)
            z_scores[valid_indices] = np.abs(stats.zscore(df_clean.loc[valid_indices, col]))
            outlier_mask = z_scores > threshold
            outlier_mask = outlier_mask.fillna(False)
            
        elif method == 'modified_zscore':
            # Modified Z-score için sadece null olmayan değerleri kullan
            valid_indices = df_clean[col].notna()
            if valid_indices.sum() < 3:
                continue
                
            valid_data = df_clean.loc[valid_indices, col]
            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))
            
            if mad == 0:  # MAD sıfır ise skip et
                continue
                
            modified_z_scores = pd.Series(index=df_clean.index, dtype=float)
            modified_z_scores[valid_indices] = 0.6745 * (valid_data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_mask = outlier_mask.fillna(False)
        
        # Outlier'ları çıkar
        outliers_before = outlier_mask.sum()
        df_clean = df_clean[~outlier_mask]
        outliers_removed = original_count - len(df_clean)
        removed_count += outliers_removed
        
        outlier_summary[col] = {
            'outliers_detected': outliers_before,
            'outliers_removed': outliers_removed,
            'percentage_removed': (outliers_removed / original_count) * 100
        }
        
        if outliers_removed > 0:
            print(f"   {col}: {outliers_removed} outlier çıkarıldı ({(outliers_removed/original_count)*100:.2f}%)")
    
    print(f"\n✅ Toplam {removed_count} outlier çıkarıldı")
    print(f"📊 Orijinal boyut: {len(df):,} → Temizlenmiş boyut: {len(df_clean):,}")
    print(f"🎯 Veri kaybı: {((len(df) - len(df_clean)) / len(df)) * 100:.2f}%")
    
    return df_clean, outlier_summary

def enhanced_correlation_analysis(df):
    """Gelişmiş korelasyon analizi - düşük korelasyonları da dahil et"""
    print(f"\n🔍 GELİŞMİŞ KORELASYON ANALİZİ")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("❌ En az 2 sayısal değişken gerekli!")
        return {}
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Korelasyon kategorileri
    correlation_categories = {
        'very_high': [],      # |r| > 0.8
        'high': [],           # 0.6 < |r| <= 0.8
        'moderate': [],       # 0.3 < |r| <= 0.6
        'low': [],            # 0.1 < |r| <= 0.3
        'very_low': []        # |r| <= 0.1
    }
    
    # Tüm korelasyon çiftlerini kategorize et
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            abs_corr = abs(corr_val)
            
            pair_info = (var1, var2, corr_val)
            
            if abs_corr > 0.8:
                correlation_categories['very_high'].append(pair_info)
            elif abs_corr > 0.6:
                correlation_categories['high'].append(pair_info)
            elif abs_corr > 0.3:
                correlation_categories['moderate'].append(pair_info)
            elif abs_corr > 0.1:
                correlation_categories['low'].append(pair_info)
            else:
                correlation_categories['very_low'].append(pair_info)
    
    # Sonuçları göster
    print(f"📊 Toplam {len(numeric_cols)} değişken arasında {len(numeric_cols)*(len(numeric_cols)-1)//2} çift analiz edildi:\n")
    
    if correlation_categories['very_high']:
        print("🔴 ÇOK YÜKSEK Korelasyonlar (|r| > 0.8) - Çoklu bağıntı riski:")
        for var1, var2, corr in correlation_categories['very_high']:
            print(f"   {var1} ↔ {var2}: r = {corr:.3f}")
    
    if correlation_categories['high']:
        print("\n🟠 YÜKSEK Korelasyonlar (0.6 < |r| ≤ 0.8):")
        for var1, var2, corr in correlation_categories['high']:
            print(f"   {var1} ↔ {var2}: r = {corr:.3f}")
    
    if correlation_categories['moderate']:
        print("\n🟡 ORTA Korelasyonlar (0.3 < |r| ≤ 0.6):")
        for var1, var2, corr in correlation_categories['moderate']:
            print(f"   {var1} ↔ {var2}: r = {corr:.3f}")
    
    if correlation_categories['low']:
        print("\n🟢 DÜŞÜK Korelasyonlar (0.1 < |r| ≤ 0.3) - Dikkat edilmesi gerekenler:")
        for var1, var2, corr in correlation_categories['low']:
            print(f"   {var1} ↔ {var2}: r = {corr:.3f}")
    
    # Uber verisi için domain spesifik mantık kontrolleri
    print(f"\n🚗 UBER VERİSİ ÖZEL KONTROLLER:")
    print("-" * 30)
    
    uber_logical_pairs = [
        ('distance', 'duration', 'Mesafe-Süre pozitif korelasyon beklenir'),
        ('distance', 'fare', 'Mesafe-Ücret pozitif korelasyon beklenir'),
        ('duration', 'fare', 'Süre-Ücret pozitif korelasyon beklenir'),
        ('distance', 'tip', 'Mesafe-Bahşiş zayıf pozitif olabilir'),
        ('fare', 'tip', 'Ücret-Bahşiş pozitif korelasyon beklenir'),
        ('rating', 'tip', 'Değerlendirme-Bahşiş pozitif korelasyon beklenir')
    ]
    
    domain_issues = []
    for var1_pattern, var2_pattern, explanation in uber_logical_pairs:
        # Sütun isimlerinde pattern ara
        var1_matches = [col for col in numeric_cols if var1_pattern.lower() in col.lower()]
        var2_matches = [col for col in numeric_cols if var2_pattern.lower() in col.lower()]
        
        for v1 in var1_matches:
            for v2 in var2_matches:
                if v1 != v2 and v1 in correlation_matrix.columns and v2 in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[v1, v2]
                    
                    # Mantık kontrolleri
                    if 'pozitif' in explanation and corr_val < 0:
                        domain_issues.append(f"⚠️  {v1} ↔ {v2}: r={corr_val:.3f} (Negatif, ama {explanation})")
                    elif 'pozitif' in explanation and abs(corr_val) < 0.1:
                        domain_issues.append(f"💡 {v1} ↔ {v2}: r={corr_val:.3f} (Çok zayıf, {explanation})")
                    elif abs(corr_val) > 0.3:
                        print(f"✅ {v1} ↔ {v2}: r={corr_val:.3f} (Beklenen - {explanation})")
    
    if domain_issues:
        print("\n🔍 Mantık dışı korelasyonlar:")
        for issue in domain_issues:
            print(f"   {issue}")
    else:
        print("✅ Domain mantığı açısından anormal korelasyon tespit edilmedi")
    
    return correlation_categories

def compare_before_after_cleaning(df_original, df_clean, outlier_summary):
    """Temizleme öncesi ve sonrası karşılaştırma"""
    print(f"\n📊 TEMİZLEME ÖNCESİ VS SONRASI KARŞILAŞTIRMA")
    print("="*60)
    
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    
    comparison_results = {}
    
    print(f"📈 İstatistiksel Değişimler:")
    print("-" * 40)
    
    for col in numeric_cols:
        if col in df_clean.columns:
            orig_data = df_original[col].dropna()
            clean_data = df_clean[col].dropna()
            
            if len(orig_data) > 0 and len(clean_data) > 0:
                # İstatistikler
                orig_mean = orig_data.mean()
                clean_mean = clean_data.mean()
                orig_std = orig_data.std()
                clean_std = clean_data.std()
                orig_median = orig_data.median()
                clean_median = clean_data.median()
                
                # Değişim oranları - sıfıra bölme kontrolü
                mean_change = ((clean_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
                std_change = ((clean_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
                median_change = ((clean_median - orig_median) / orig_median) * 100 if orig_median != 0 else 0
                
                print(f"\n📊 {col}:")
                print(f"   Ortalama: {orig_mean:.2f} → {clean_mean:.2f} ({mean_change:+.1f}%)")
                print(f"   Std Sapma: {orig_std:.2f} → {clean_std:.2f} ({std_change:+.1f}%)")
                print(f"   Medyan: {orig_median:.2f} → {clean_median:.2f} ({median_change:+.1f}%)")
                
                comparison_results[col] = {
                    'mean_change_pct': mean_change,
                    'std_change_pct': std_change,
                    'median_change_pct': median_change,
                    'outliers_removed': outlier_summary.get(col, {}).get('outliers_removed', 0)
                }
    
    # Genel değerlendirme
    total_outliers = sum(info.get('outliers_removed', 0) for info in outlier_summary.values())
    data_loss_pct = ((len(df_original) - len(df_clean)) / len(df_original)) * 100
    
    print(f"\n🎯 GENEL DEĞERLENDİRME:")
    print(f"   Toplam outlier çıkarıldı: {total_outliers:,}")
    print(f"   Veri kaybı: {data_loss_pct:.2f}%")
    
    if data_loss_pct < 5:
        print("   ✅ Kabul edilebilir veri kaybı")
    elif data_loss_pct < 10:
        print("   ⚠️  Orta seviye veri kaybı - dikkatli ol")
    else:
        print("   ❌ Yüksek veri kaybı - stratejini gözden geçir")
    
    return comparison_results

def create_cleaning_workflow(df, methods=['iqr'], thresholds=[1.5]):
    """Outlier temizleme iş akışı - farklı yöntemleri dene"""
    print(f"\n🔄 OUTLIER TEMİZLEME İŞ AKIŞI")
    print("="*60)
    
    results = {}
    
    for method in methods:
        for threshold in thresholds:
            print(f"\n📋 {method.upper()} yöntemi, eşik: {threshold}")
            print("-" * 40)
            
            df_cleaned, outlier_summary = clean_outliers(df, method=method, threshold=threshold)
            
            # Bu temizleme için güvenirlik testini tekrarla
            reliability_results = statistical_reliability_tests(df_cleaned)
            
            # Karşılaştırma
            comparison = compare_before_after_cleaning(df, df_cleaned, outlier_summary)
            
            results[f"{method}_{threshold}"] = {
                'cleaned_df': df_cleaned,
                'outlier_summary': outlier_summary,
                'reliability_results': reliability_results,
                'comparison': comparison,
                'data_loss_pct': ((len(df) - len(df_cleaned)) / len(df)) * 100
            }
    
    # En iyi yöntemi öner
    print(f"\n🏆 YÖNTEM ÖNERİSİ:")
    print("-" * 30)
    
    best_method = None
    best_score = -1
    
    for method_name, result in results.items():
        # Skor hesapla: düşük veri kaybı + yüksek güvenirlik
        data_loss = result['data_loss_pct']
        
        # Basit skorlama
        if data_loss < 5:
            loss_score = 10
        elif data_loss < 10:
            loss_score = 7
        elif data_loss < 15:
            loss_score = 4
        else:
            loss_score = 1
            
        total_score = loss_score
        
        if total_score > best_score:
            best_score = total_score
            best_method = method_name
        
        print(f"   {method_name}: Veri kaybı {data_loss:.1f}%, Skor: {total_score}")
    
    print(f"\n💡 ÖNERİLEN: {best_method} yöntemi")
    
    return results, best_method

def create_reliability_visualizations(df, results):
    """Güvenirlik analizi görselleştirmeleri"""
    print("\n📊 GÜVENİRLİK ANALİZİ GÖRSELLEŞTİRMELERİ")
    print("-" * 50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 1. Normallik test sonuçları
    if 'normality' in results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Normallik Test Sonuçları', fontsize=16)
        
        for i, col in enumerate(numeric_cols[:4]):
            if col in results['normality']:
                row, col_idx = i // 2, i % 2
                ax = axes[row, col_idx]
                
                data = df[col].dropna()
                
                # Q-Q Plot
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'{col} - Q-Q Plot')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/mertcagatay/Desktop/BitirmeDenemeler/normality_tests.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Korelasyon ısı haritası
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Değişkenler Arası Korelasyon Matrisi')
        plt.tight_layout()
        plt.savefig('/Users/mertcagatay/Desktop/BitirmeDenemeler/correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Outlier görselleştirmesi
    if len(numeric_cols) > 0:
        n_cols = min(len(numeric_cols), 4)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            ax = axes[i]
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'{col}\nOutlier Analizi')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/mertcagatay/Desktop/BitirmeDenemeler/outlier_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✅ Güvenirlik analizi görselleştirmeleri oluşturuldu")

def generate_reliability_report(df, results):
    """Güvenirlik analizi raporu oluştur"""
    print("\n" + "="*80)
    print("GÜVENİRLİK ANALİZİ RAPORU")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\n📊 VERİ SETİ ÖZETİ:")
    print(f"   - Toplam gözlem: {len(df):,}")
    print(f"   - Sayısal değişken: {len(numeric_cols)}")
    print(f"   - Eksik değer oranı: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    reliability_score = 0
    max_score = 0
    
    # Normallik skoru
    if 'normality' in results and results['normality']:
        normal_count = sum(1 for r in results['normality'].values() if r['is_normal'])
        normality_score = (normal_count / len(results['normality'])) * 25
        reliability_score += normality_score
        print(f"\n✅ NORMALLİK SKORU: {normality_score:.1f}/25")
        print(f"   {normal_count}/{len(results['normality'])} değişken normal dağılımlı")
    max_score += 25
    
    # Outlier skoru
    if 'outliers' in results and results['outliers']:
        good_outlier_count = sum(1 for r in results['outliers'].values() 
                               if r['average_percentage'] < 5)
        outlier_score = (good_outlier_count / len(results['outliers'])) * 25
        reliability_score += outlier_score
        print(f"\n✅ OUTLIER SKORU: {outlier_score:.1f}/25")
        print(f"   {good_outlier_count}/{len(results['outliers'])} değişken kabul edilebilir outlier seviyesinde")
    max_score += 25
    
    # Korelasyon skoru
    correlation_score = 25
    if 'high_correlations' in results:
        high_corr_count = len(results['high_correlations'])
        if high_corr_count > 0:
            correlation_score = max(0, 25 - (high_corr_count * 5))
        print(f"\n✅ KORELASYON SKORU: {correlation_score:.1f}/25")
        print(f"   {high_corr_count} yüksek korelasyon çifti tespit edildi")
    reliability_score += correlation_score
    max_score += 25
    
    # Tutarlılık skoru
    consistency_score = 25
    if 'consistency_issues' in results:
        issue_count = len(results['consistency_issues'])
        if issue_count > 0:
            consistency_score = max(0, 25 - (issue_count * 3))
        print(f"\n✅ TUTARLILIK SKORU: {consistency_score:.1f}/25")
        print(f"   {issue_count} tutarlılık sorunu tespit edildi")
    reliability_score += consistency_score
    max_score += 25
    
    # Genel değerlendirme
    final_score = (reliability_score / max_score) * 100
    
    print(f"\n" + "="*50)
    print(f"🎯 GENEL GÜVENİRLİK SKORU: {final_score:.1f}/100")
    
    if final_score >= 85:
        print("🌟 MÜKEMMEL - Veri seti yüksek güvenirlikte")
        recommendation = "Veri seti doğrudan analiz için uygun"
    elif final_score >= 70:
        print("✅ İYİ - Veri seti güvenilir seviyede") 
        recommendation = "Minimal temizleme ile kullanılabilir"
    elif final_score >= 50:
        print("⚠️  ORTA - Dikkatli kullanım gerekli")
        recommendation = "Kapsamlı veri temizleme önerilir"
    else:
        print("❌ DÜŞÜK - Ciddi veri kalitesi sorunları")
        recommendation = "Yoğun veri ön işleme gerekli"
    
    print(f"💡 ÖNERİ: {recommendation}")
    
    return {
        'final_score': final_score,
        'recommendation': recommendation,
        'detailed_scores': {
            'normality': normality_score if 'normality' in results else 0,
            'outliers': outlier_score if 'outliers' in results else 0,
            'correlation': correlation_score,
            'consistency': consistency_score
        }
    }

def main():
    """Main analysis function"""
    print("Python Uber Data Analysis - Starting Analysis...")
    print("="*60)
    
    # Veri setini indir
    dataset_path = download_dataset()
    
    if dataset_path:
        # Veri setini yükle ve keşfet
        result = load_and_explore_data(dataset_path)
        
        if result and result[0] is not None:
            df, csv_files = result
            
            print(f"\n{'='*80}")
            print("📊 AŞAMA 1: TAM VERİ SETİ İLE KAPSAMLI ANALİZ")
            print("="*80)
            
            # 1.1 Temel veri analizi
            print(f"\n🔍 1.1 - Temel Veri Analizi")
            print("-" * 50)
            df = basic_data_analysis(df)
            
            # 1.2 İlk görselleştirmeler
            print(f"\n📊 1.2 - Temel Görselleştirmeler")
            print("-" * 50)
            create_initial_visualizations(df)
            
            # 1.3 İstatistiksel güvenirlik testleri (TAM VERİ)
            print(f"\n🔬 1.3 - İstatistiksel Güvenirlik Testleri (TAM VERİ)")
            print("-" * 50)
            reliability_results_full = statistical_reliability_tests(df)
            
            # 1.4 Güvenirlik görselleştirmeleri (TAM VERİ)
            print(f"\n📈 1.4 - Güvenirlik Görselleştirmeleri (TAM VERİ)")
            print("-" * 50)
            create_reliability_visualizations(df, reliability_results_full)
            
            # 1.5 Güvenirlik raporu (TAM VERİ)
            print(f"\n📋 1.5 - Güvenirlik Raporu (TAM VERİ)")
            print("-" * 50)
            final_report_full = generate_reliability_report(df, reliability_results_full)
            
            # 1.6 Gelişmiş korelasyon analizi (TAM VERİ)
            print(f"\n🔗 1.6 - Gelişmiş Korelasyon Analizi (TAM VERİ)")
            print("-" * 50)
            correlation_categories_full = enhanced_correlation_analysis(df)
            
            print(f"\n✅ AŞAMA 1 TAMAMLANDI - TAM VERİ ANALİZİ")
            print(f"📊 Güvenirlik Skoru: {final_report_full['final_score']:.1f}/100")
            
            # =====================================================
            
            print(f"\n{'='*80}")
            print("🧹 AŞAMA 2: OUTLIER TEMİZLEME VE YENİDEN ANALİZ")
            print("="*80)
            
            # 2.1 Outlier temizleme iş akışı
            print(f"\n🔄 2.1 - Outlier Temizleme İş Akışı")
            print("-" * 50)
            cleaning_results, best_method = create_cleaning_workflow(
                df, 
                methods=['iqr', 'zscore', 'modified_zscore'], 
                thresholds=[1.5, 2.0, 2.5]
            )
            
            # En iyi temizlenmiş veri setini seç
            best_cleaned_df = cleaning_results[best_method]['cleaned_df']
            
            print(f"\n🏆 En İyi Temizleme Yöntemi: {best_method}")
            print(f"📉 Veri Kaybı: {cleaning_results[best_method]['data_loss_pct']:.2f}%")
            
            # 2.2 Temizlenmiş veri ile güvenirlik testleri
            print(f"\n🔬 2.2 - İstatistiksel Güvenirlik Testleri (TEMİZLENMİŞ VERİ)")
            print("-" * 50)
            reliability_results_clean = statistical_reliability_tests(best_cleaned_df)
            
            # 2.3 Temizlenmiş veri ile güvenirlik raporu
            print(f"\n📋 2.3 - Güvenirlik Raporu (TEMİZLENMİŞ VERİ)")
            print("-" * 50)
            final_report_clean = generate_reliability_report(best_cleaned_df, reliability_results_clean)
            
            # 2.4 Temizlenmiş veri ile korelasyon analizi
            print(f"\n🔗 2.4 - Gelişmiş Korelasyon Analizi (TEMİZLENMİŞ VERİ)")
            print("-" * 50)
            correlation_categories_clean = enhanced_correlation_analysis(best_cleaned_df)
            
            # 2.5 Karşılaştırmalı sonuçlar
            print(f"\n⚖️  2.5 - Öncesi vs Sonrası Karşılaştırma")
            print("-" * 50)
            comparison_results = compare_before_after_cleaning(df, best_cleaned_df, cleaning_results[best_method]['outlier_summary'])
            
            print(f"\n{'='*80}")
            print("📈 AŞAMA 3: KAPSAMLI KARŞILAŞTIRMA VE SONUÇ RAPORU")
            print("="*80)
            
            # 3.1 Güvenirlik skoru karşılaştırması
            print(f"\n📊 3.1 - Güvenirlik Skoru Karşılaştırması")
            print("-" * 50)
            print(f"TAM VERİ Güvenirlik Skoru: {final_report_full['final_score']:.1f}/100")
            print(f"TEMİZLENMİŞ VERİ Güvenirlik Skoru: {final_report_clean['final_score']:.1f}/100")
            improvement = final_report_clean['final_score'] - final_report_full['final_score']
            if improvement > 0:
                print(f"🎯 İYİLEŞME: +{improvement:.1f} puan (✅ Outlier temizleme etkili)")
            elif improvement < 0:
                print(f"⚠️  DÜŞÜŞ: {improvement:.1f} puan (Dikkatli kullanım gerekli)")
            else:
                print(f"➡️  DEĞİŞİM YOK: Outlier temizleme etkisiz")
            
            # 3.2 Korelasyon değişimleri
            print(f"\n🔗 3.2 - Korelasyon Değişimleri")
            print("-" * 50)
            
            def count_correlations(categories):
                return {
                    'very_high': len(categories.get('very_high', [])),
                    'high': len(categories.get('high', [])),
                    'moderate': len(categories.get('moderate', [])),
                    'low': len(categories.get('low', [])),
                    'very_low': len(categories.get('very_low', []))
                }
            
            corr_before = count_correlations(correlation_categories_full)
            corr_after = count_correlations(correlation_categories_clean)
            
            print("Korelasyon Seviyesi Değişimleri:")
            for level in ['very_high', 'high', 'moderate', 'low', 'very_low']:
                before = corr_before[level]
                after = corr_after[level]
                change = after - before
                change_str = f"{change:+d}" if change != 0 else "0"
                print(f"   {level.replace('_', ' ').title()}: {before} → {after} ({change_str})")
            
            # Veri setlerini global değişken olarak sakla
            globals()['uber_df'] = df  # Orijinal veri
            globals()['uber_df_clean'] = best_cleaned_df  # Temizlenmiş veri
            globals()['dataset_files'] = csv_files
            
            # Tam veri sonuçları
            globals()['reliability_results_full'] = reliability_results_full
            globals()['reliability_report_full'] = final_report_full
            globals()['correlation_categories_full'] = correlation_categories_full
            
            # Temizlenmiş veri sonuçları
            globals()['reliability_results_clean'] = reliability_results_clean
            globals()['reliability_report_clean'] = final_report_clean
            globals()['correlation_categories_clean'] = correlation_categories_clean
            
            # Temizleme bilgileri
            globals()['cleaning_results'] = cleaning_results
            globals()['best_cleaning_method'] = best_method
            globals()['comparison_results'] = comparison_results
            
            print(f"\n{'='*80}")
            print("✅ KAPSAMLI 2-AŞAMALI ANALİZ TAMAMLANDI!")
            print("="*80)
            print(f"\n📊 VERİ SETLERİ:")
            print(f"   🗂️  Orijinal: 'uber_df' ({len(df):,} satır)")
            print(f"   🧹 Temizlenmiş: 'uber_df_clean' ({len(best_cleaned_df):,} satır)")
            print(f"   📁 Diğer dosyalar: {[f.name for f in csv_files]}")
            
            print(f"\n📈 GÜVENİRLİK SKORLARI:")
            print(f"   🔴 Tam Veri: {final_report_full['final_score']:.1f}/100")
            print(f"   🟢 Temizlenmiş: {final_report_clean['final_score']:.1f}/100")
            print(f"   📊 İyileşme: {improvement:+.1f} puan")
            
            print(f"\n🧹 TEMİZLEME BİLGİLERİ:")
            print(f"   🏆 En iyi yöntem: {best_method}")
            print(f"   📉 Veri kaybı: {cleaning_results[best_method]['data_loss_pct']:.2f}%")
            
            print(f"\n💡 SONUÇ: ")
            if improvement > 5:
                print("   🌟 Outlier temizleme önemli iyileşme sağladı!")
            elif improvement > 0:
                print("   ✅ Outlier temizleme olumlu etki gösterdi.")
            else:
                print("   ⚠️  Outlier temizleme dikkatli değerlendirilmeli.")
            
            print(f"\n🎯 Her iki veri seti de analizleriniz için hazır!")
        
        else:
            print("❌ Veri yükleme başarısız!")
    else:
        print("❌ Veri seti indirme başarısız!")

if __name__ == "__main__":
    main()
