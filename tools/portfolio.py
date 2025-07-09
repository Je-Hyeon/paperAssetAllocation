import numpy as np
import pandas as pd

def black_litterman(
    sigma,        # 공분산 행렬 (N×N)
    w_mkt,        # 시장 비중 벡터 (N,)
    p,            # 뷰 매핑 행렬 (K×N)
    q,            # 뷰 기대 수익 벡터 (K,)
    omega,        # 뷰 분산 행렬 (K×K)
    delta:float=2.5,    # 위험회피계수
    tau:float=1      # π 불확실도 스케일
):
    pi = delta * sigma.dot(w_mkt)

    tau_Sigma_inv = np.linalg.inv(tau * sigma)
    Omega_inv     = np.linalg.inv(omega)
    
    M = tau_Sigma_inv + p.T.dot(Omega_inv).dot(p)
    b = tau_Sigma_inv.dot(pi) + p.T.dot(Omega_inv).dot(q)
    mu_bl = np.linalg.solve(M, b)  

    w_bl = (1.0 / delta) * np.linalg.inv(sigma).dot(mu_bl)
    return mu_bl, w_bl


# 2) 상삼각 벡터화
def vectorize_corr(mat):
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def generate_monthly_data(df:pd.DataFrame, col1='date', col2='asset'):
    df.index.set_names([col1, col2], inplace=True)

    df_monthly = (
        df
        .groupby(level=col2)
        .resample('ME', level=col1)
        .last()
        .swaplevel(0,1)
        .sort_index()
    ) 
    idx = pd.MultiIndex.from_product(
        [
            df_monthly.index.get_level_values(0).unique().tolist(), 
            df_monthly
            .index.get_level_values(1).unique().tolist()
        ]
    )
    return df_monthly.loc[
        idx,
        df_monthly.index.get_level_values(1).unique().tolist()
    ]
    

def simulate_longonly(
    group_weight_df:pd.DataFrame, 
    price_df:pd.DataFrame, 
    fee_rate:float, 
    pf_value:float=1,
    leverage:float=1
    ) -> pd.Series:
    
    pf_dict = {}
    start_idx = group_weight_df.index[0]
    rebalance_idx = group_weight_df.index   
    
    pf_dict[start_idx] = pf_value
    investment_long_weight = group_weight_df.iloc[0]
    
    investment_long_dv = investment_long_weight * (pf_value * leverage)
    
    entry_pf_value = pf_value
    entry_price = np.nan_to_num(price_df.loc[group_weight_df.index[0]]) # 매입가 기록 (공매도의 경우 매도가가 된다)
    num_shares_long = np.nan_to_num((investment_long_dv / entry_price).values) 
    
    for idx, current_price_row in price_df.loc[start_idx:].iloc[1:].iterrows(): 
        price_diff = (current_price_row - entry_price)
        long_p_and_l = (price_diff * num_shares_long)
        pf_value = entry_pf_value + np.nansum(long_p_and_l) 

        if pf_value <= 0:
            pf_value = 0
            long_p_and_l = np.zeros_like(num_shares_long)
            num_shares_long = np.zeros_like(num_shares_long)
            entry_pf_value = 0

        if idx in rebalance_idx: 
            target_long_weight = group_weight_df.loc[idx]
            target_investment_long_dv = target_long_weight * (pf_value * leverage)  # target DV
            
            # 현재 DV
            investment_long_dv = np.nan_to_num(num_shares_long) * np.nan_to_num(current_price_row)
            
            dv_delta_sell = np.abs(np.nan_to_num(target_investment_long_dv) - np.nan_to_num(investment_long_dv))
            fee = np.nansum(dv_delta_sell) * fee_rate
            pf_value -= fee
            
            entry_price = current_price_row # 진입 가격 업데이트
            entry_pf_value = pf_value
            
            investment_long_dv = target_long_weight * (pf_value*leverage) # dv 업데이트
            
            num_shares_long =  np.nan_to_num(investment_long_dv / entry_price) # 매수한 코인 개수 업데이트
            
        pf_dict[idx] = pf_value

    return pd.Series(pf_dict)