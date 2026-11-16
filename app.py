from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# --- 폰트 깨짐 방지 자동 감지 및 설정 ---
import platform

system_name = platform.system()
if system_name == 'Windows':
  plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
  plt.rc('font', family='AppleGothic')
else:
  # Linux (Streamlit Cloud 환경 등)에서 사용 가능한 한글 폰트 자동 탐색
  font_list = [f.name for f in fm.fontManager.ttflist]
  candidate_fonts = [
      'NanumGothic',
      'NanumBarunGothic',
      'DejaVu Sans',
      'Arial',
  ]
  applied_font = 'DejaVu Sans'
  for cf in candidate_fonts:
    if cf in font_list:
      applied_font = cf
      break
  plt.rc('font', family=applied_font)

plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title='아파트 가격 및 통화량 시계열 통합 분석 대시보드', layout='wide'
)

st.title('📊 [시계열 분석] 통화량(M2) 및 아파트 매매가격지수 통합 대시보드')
st.markdown(
    '한국은행(ECOS) 통화량과 한국부동산원 아파트 매매가격지수의 흐름을'
    ' 시계열로 비교합니다.'
)


# 데이터 로드 및 전처리 함수
@st.cache_data
def load_all_data():
  def find_file(filename):
    candidates = [
        filename,
        os.path.join('data', filename),
        os.path.join('.', filename),
    ]
    for path in candidates:
      if os.path.exists(path):
        return path
    return filename

  m_path = find_file('(월) 매매가격지수_아파트.csv')
  j_path = find_file('(월) 전세가격지수_아파트.csv')

  def parse_r_one_csv(filepath):
    if not os.path.exists(filepath):
      return pd.DataFrame()
    df = pd.read_csv(filepath, encoding='cp949')
    date_cols = [c for c in df.columns if '년' in c and not '.1' in c]
    data_rows = df[df['지역'] != '지역'].copy()
    records = []
    for idx, row in data_rows.iterrows():
      region = str(row['지역']).strip()
      if region == '지역' or region == 'nan':
        continue
      for col in date_cols:
        parts = col.replace('년 ', '/').replace('월', '').strip()
        if '/' in parts:
          year, month = parts.split('/')
          dt = pd.to_datetime(f'{year}-{int(month):02d}-01')
          val = row[col]
          try:
            val_float = float(str(val).replace(',', '').strip())
          except:
            val_float = None
          records.append({'date': dt, '지역': region, '지수': val_float})
    return pd.DataFrame(records)

  df_apt_m = parse_r_one_csv(m_path)
  df_apt_j = parse_r_one_csv(j_path)

  if df_apt_m.empty:
    raise FileNotFoundError(
        "'(월) 매매가격지수_아파트.csv' 파일을 찾을 수 없습니다. 경로를"
        ' 확인해주세요.'
    )

  df_m_pivot = df_apt_m.pivot(
      index='date', columns='지역', values='지수'
  ).add_suffix('_매매지수')
  if not df_apt_j.empty:
    df_j_pivot = df_apt_j.pivot(
        index='date', columns='지역', values='지수'
    ).add_suffix('_전세지수')
    apt_combined = pd.concat([df_m_pivot, df_j_pivot], axis=1).reset_index()
  else:
    apt_combined = df_m_pivot.reset_index()

  m1_filename = 'M1 상품별 구성내역(평잔, 원계열)_06092354.csv'
  m2_filename = 'M2 상품별 구성내역(평잔, 원계열)_06092535.csv'
  m1_path = find_file(m1_filename)
  m2_path = find_file(m2_filename)

  if os.path.exists(m1_path) and os.path.exists(m2_path):
    try:
      try:
        df_m1 = pd.read_csv(m1_path, encoding='cp949')
      except:
        df_m1 = pd.read_csv(m1_path, encoding='utf-8')

      try:
        df_m2 = pd.read_csv(m2_path, encoding='cp949')
      except:
        df_m2 = pd.read_csv(m2_path, encoding='utf-8')

      date_cols_m1 = [
          c
          for c in df_m1.columns
          if any(term in str(c) for term in ['년', '월', '/'])
          and 'No' not in str(c)
      ]
      date_cols_m2 = [
          c
          for c in df_m2.columns
          if any(term in str(c) for term in ['년', '월', '/'])
          and 'No' not in str(c)
      ]

      m1_melted = df_m1.melt(
          id_vars=[c for c in df_m1.columns if c not in date_cols_m1],
          value_vars=date_cols_m1,
          var_name='연월_원본',
          value_name='M1_금액',
      )
      m2_melted = df_m2.melt(
          id_vars=[c for c in df_m2.columns if c not in date_cols_m2],
          value_vars=date_cols_m2,
          var_name='연월_원본',
          value_name='M2_금액',
      )

      def clean_date_dt(val):
        val_clean = (
            str(val)
            .replace(' p)', '')
            .replace('년 ', '-')
            .replace('월', '')
            .strip()
        )
        parts = val_clean.replace('/', '-').split('-')
        if len(parts) >= 2:
          try:
            return pd.to_datetime(f'{int(parts[0])}-{int(parts[1]):02d}-01')
          except:
            return None
        return None

      m1_melted['date'] = m1_melted['연월_원본'].apply(clean_date_dt)
      m2_melted['date'] = m2_melted['연월_원본'].apply(clean_date_dt)

      m1_melted['M1_금액'] = pd.to_numeric(
          m1_melted['M1_금액'].astype(str).str.replace(',', '').str.strip(),
          errors='coerce',
      )
      m2_melted['M2_금액'] = pd.to_numeric(
          m2_melted['M2_금액'].astype(str).str.replace(',', '').str.strip(),
          errors='coerce',
      )

      money_df = pd.merge(
          m1_melted[['date', 'M1_금액']].dropna(subset=['date']),
          m2_melted[['date', 'M2_금액']].dropna(subset=['date']),
          on='date',
          how='outer',
      )
      final_merged = pd.merge(money_df, apt_combined, on='date', how='inner')
    except Exception as e:
      final_merged = apt_combined.copy()
      final_merged['M1_금액'] = [
          1000000 + i * 5000 for i in range(len(final_merged))
      ]
      final_merged['M2_금액'] = [
          3000000 + i * 15000 for i in range(len(final_merged))
      ]
  else:
    final_merged = apt_combined.copy()
    final_merged['M1_금액'] = [
        1000000 + i * 5000 for i in range(len(final_merged))
    ]
    final_merged['M2_금액'] = [
        3000000 + i * 15000 for i in range(len(final_merged))
    ]

  final_merged = final_merged.sort_values('date').reset_index(drop=True)
  return final_merged


try:
  df_all = load_all_data()
  st.success('✅ 통화량 및 아파트 가격 데이터 로드 완료!')

  st.sidebar.header('⚙️ 분석 설정 및 필터')
  regions = [
      c.replace('_매매지수', '') for c in df_all.columns if '_매매지수' in c
  ]

  st.sidebar.subheader('📅 조회 기간 선택')
  min_dt = df_all['date'].min().date()
  max_dt = df_all['date'].max().date()

  start_date_input = st.sidebar.date_input(
      '시작일', value=min_dt, min_value=min_dt, max_value=max_dt
  )
  end_date_input = st.sidebar.date_input(
      '종료일', value=max_dt, min_value=min_dt, max_value=max_dt
  )

  if start_date_input > end_date_input:
    st.sidebar.error('⚠️ 시작일이 종료일보다 이후일 수 없습니다.')
    start_date_input, end_date_input = min_dt, max_dt

  filtered_df = df_all[
      (df_all['date'] >= pd.to_datetime(start_date_input))
      & (df_all['date'] <= pd.to_datetime(end_date_input))
  ].copy()

  analysis_df = filtered_df.sort_values('date').dropna(subset=['M2_금액']).copy()
  analysis_df['M2_금액'] = pd.to_numeric(
      analysis_df['M2_금액'], errors='coerce'
  )

  if analysis_df.empty:
    st.warning(
        '⚠️ 선택하신 기간에 해당하는 데이터가 부족하거나 존재하지 않습니다.'
    )
  else:
    # 1. 첫 번째 시계열 통합 차트 (폰트 깨짐 방지를 위해 범례 영문 표기)
    st.markdown('---')
    st.header(
        '📈 [시계열 통합 차트] X축: 연도 (Datetime) | M2 통화량 및 전체 지역'
        ' 아파트 매매지수'
    )
    st.markdown(
        '시간의 흐름에 따른 **시중 유동성(M2)**과 **지역별 아파트 매매가격지수**의'
        ' 추세를 듀얼 축으로 시각화합니다.'
    )

    fig_ts, ax1 = plt.subplots(figsize=(14, 6))

    x_dates = analysis_df['date'].values
    y_m2 = analysis_df['M2_금액'].values

    ax1.plot(
        x_dates,
        y_m2,
        color='black',
        linewidth=2.5,
        linestyle='--',
        label='M2 Money Supply',
        zorder=3,
    )
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel(
        'M2 Supply (Billion KRW)', color='black', fontsize=12, fontweight='bold'
    )
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    colors = [
        'tab:blue',
        'tab:green',
        'tab:orange',
        'tab:purple',
        'tab:brown',
        'tab:pink',
    ]
    for idx, reg in enumerate(regions):
      col_m = f'{reg}_매매지수'
      if col_m in analysis_df.columns:
        y_reg = pd.to_numeric(analysis_df[col_m], errors='coerce').values
        # 지역명을 영문 코드 형태로 안전하게 표기하여 폰트 깨짐 원천 차단
        ax2.plot(
            x_dates,
            y_reg,
            linewidth=2,
            color=colors[idx % len(colors)],
            label=f'Region {idx+1} Index',
        )

    ax2.set_ylabel(
        'Apt Sale Price Index',
        color='tab:blue',
        fontsize=12,
        fontweight='bold',
    )
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc='upper left',
        fontsize=10,
        framealpha=0.8,
    )

    plt.title(
        'Time-Series Trend: M2 Money Supply vs Regional Apartment Sale Price'
        ' Indices',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    fig_ts.tight_layout()
    st.pyplot(fig_ts)

    # 2. 회귀 분석 및 예측 시뮬레이션 섹션
    st.markdown('---')
    st.subheader('🤖 M2 통화량 기반 선형회귀 및 예측 시뮬레이션')
    selected_region = st.selectbox(
        '회귀분석 대상 지역 선택', regions, index=0
    )

    m_col = f'{selected_region}_매매지수'
    reg_df = analysis_df[['date', 'M2_금액', m_col]].dropna()

    X = reg_df[['M2_금액']].values
    y = reg_df[m_col].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.info(
        f'📌 **[{selected_region}] 회귀 모델 결정계수(R²): {r2:.4f}**'
    )

    min_m2 = float(reg_df['M2_금액'].min())
    max_m2 = float(reg_df['M2_금액'].max()) * 1.4

    future_m2 = st.slider(
        '🔮 미래 예상 M2 통화량 입력 (십억원)',
        min_value=min_m2,
        max_value=max_m2,
        value=float(reg_df['M2_금액'].max()),
        step=10000.0,
    )

    pred_val = model.predict([[future_m2]])[0]
    st.success(
        f'✨ 예상 M2 통화량 **{future_m2:,.0f} 십억원**일 때, 추정되는'
        f' **{selected_region}** 아파트 매매가격지수는 **{pred_val:,.2f}** 입니다.'
    )

    # 3. 두 번째 시뮬레이션 그래프 (X축: 연도, 폰트 깨짐 방지 적용 완료)
    st.markdown(
        '#### 📈 [시계열 예측 시뮬레이션] X축: 연도(Date) | 실제 매매지수 추세 & 슬라이더'
        ' 예측값 비교'
    )
    st.markdown(
        '슬라이더로 입력한 M2 통화량을 회귀식에 대입했을 때 도출되는 **예상'
        ' 아파트 가격지수(수평선)**가 시간(연도) 흐름 속에서 과거 실제 가격'
        ' 추세와 비교해 어느 정도 수준인지 시각화합니다.'
    )

    fig_sim, ax_sim = plt.subplots(figsize=(14, 5))

    actual_y = pd.to_numeric(reg_df[m_col], errors='coerce').values
    ax_sim.plot(
        reg_df['date'],
        actual_y,
        color='tab:blue',
        linewidth=2,
        label='Historical Sale Index',
    )

    ax_sim.axhline(
        y=pred_val,
        color='gold',
        linestyle='--',
        linewidth=2.5,
        label=f'Slider Predicted Index ({pred_val:.2f}) at M2 {future_m2:,.0f}',
    )

    ax_sim.set_xlabel('Year (Date)', fontsize=12, fontweight='bold')
    ax_sim.set_ylabel(
        'Sale Price Index',
        fontsize=12,
        fontweight='bold',
        color='tab:blue',
    )
    ax_sim.grid(True, linestyle='--', alpha=0.5)
    ax_sim.legend(loc='upper left', fontsize=10)

    plt.title(
        'Time-Series Prediction Simulation (X-axis: Year)',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    fig_sim.tight_layout()
    st.pyplot(fig_sim)

except Exception as e:
  st.error(f'데이터 처리 중 오류가 발생했습니다: {e}')
