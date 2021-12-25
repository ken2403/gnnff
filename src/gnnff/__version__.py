__version_info__ = (0, 2, 3)
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
- (0.1.0)
    - res nets なくす
    - edge_jk を計算に含める
    - 学習時のメモリーサイズを小さくするために、変数名変更
- (0.2.0)
    - edge_jkなくす
    - nbr_maskをnode_kとc3に適用
- (0.2.1)
    - (NodeUpdate)nbr_maskをc1から外し、sumの前に掛け算で適用(spk参考)
    - (EdgeUpdate)nbr_maskをc2から外し、node_jのみに適用
    - (EdgeUpdate)nbr_maskをc3から外し、node_jとnode_kのみに適用
    - (GraphtoFeature)initial_embeddingに対してnbrmaskを外す
    - (get_node_k)indexのやり方変更、nbr_maskをindexを利用して適用
- (0.2.2)
    - implementation of simple prediction.
- (0.2.3)
    - remove simple implementation (because of bad result)
    - add "trainble_gaussian" parameter.
"""
