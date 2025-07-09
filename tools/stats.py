import numpy as np
import pandas as pd 

def print_statistics(return_dict:dict,
                     start_date=None):
    '''return_dict : dict(전략 수익률이 담긴 딕셔너리)
       mkt_rtn     : pd.Series (마켓 수익률이 담긴 시리즈)
       
       Note) mean, std, cagr 계산할 때 시작일은 포함하지 않습니다(시작일 수익은 0이라서)'''

    mean = []
    std = []
    mdd = []
    cum = []
    key_list = []

    for key, df in return_dict.items():
        key_list.append(key)
        
        if start_date != None:
            df2 = df.loc[start_date:].iloc[1:]
        else:
            df2 = df.iloc[1:]
        
        df2 = df.replace(0, np.nan).dropna() # 청산된 경우는 계산에서 제외...
        
        m = (df2.mean() * 365).round(5)   
        mean.append(m)
        s = (df2.std() * np.sqrt(365))
        std.append(s)
        #ca = calculate_cagr(df2)
        #cagr.append(ca)
        
        cum_df = (df2+1).cumprod()
        peak = cum_df.cummax()
        drawdown = (cum_df-peak)/peak
        mdd.append(round((-drawdown).max(),5))
        
        cu = (cum_df.iloc[-1] - 1)
        cum.append(cu)
        
    col = [key for key, df in return_dict.items()]
    return_df = pd.DataFrame([mean,std,mdd,cum], index=["Mean","Std","Mdd",'Cum'], columns=col)  
    return_df.loc["Sharpe",:] = (return_df.loc["Mean",:]) / (return_df.loc["Std",:])
    return_df = return_df.loc[['Mean', 'Std', 'Sharpe', 'Cum', 'Mdd'], :].T
    return_df.loc[:, ['Mean', 'Std', 'Cum', 'Mdd']] = return_df.loc[:, ['Mean', 'Std', 'Cum', 'Mdd']] * 100
    return return_df