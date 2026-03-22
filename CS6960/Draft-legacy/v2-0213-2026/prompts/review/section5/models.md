✅ 最終提案（バランス・実装・理論すべて考慮）
🟦 CNN Family（PANNs 系で統一）

理由：

同一論文系統

明確なパラメータ差

pretrained 公開済み

audio tagging → embedding 抽出が安定

① CNN-Small

CNN6
約 4.8M parameters

② CNN-Base

ResNet22 (PANNs)
約 63.7M parameters

③ CNN-Large

ResNet54 (PANNs)
約 104.3M parameters

CNN14（80M）も候補ですが、large を「できるだけ大きく」に寄せるなら ResNet54 の方が明確に“上限”感があります。


🟥 Transformer Family（SSAST + MERT）

Transformer は「段階的拡張」と「極大モデル」を両立させます。

① Transformer-Small

SSAST-small
約 23M

② Transformer-Base

SSAST-base
約 89M

③ Transformer-Large

MERT-330M
330M