__version_info__ = (0, 0, 5)
__version__ = ".".join(map(str, __version_info__))

"""
- (0.0.1)
    - edge_jkは無しで実装
- (0.0.2)
    - neighborのない箇所の距離を100に設定する(gaussian filterで値が0になり、以降の学習に使用されなくなる)
- (0.0.3)
    - neighborのない箇所の距離を0に戻す(loaderとの差異をなくすため)
    - nbr_maskをupdated_edgeに適用することで学習の際の負担を減らす
- (0.0.4)
    - うまく学習できないので,　'nbr_maskをupdated_edgeに適用すること'　これをやめる
- (0.0.5)
    - batchnorm を減らす
    - 毎回のmessageにresidual netを追加
    - gnnffで力とエネルギーを同時に学習できるようにレイアウトを変更
    - training時のtensor計算の最適化
"""
