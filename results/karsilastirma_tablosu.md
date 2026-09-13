| Deney | Aciklama | Model | Aug | Poz | Test Acc | Top-3 | Params | Sure(s) |
|---|---|---|---|---|---|---|---|---|
| baseline | LSTM (final model) | LSTM | VAR | VAR | %96.47 | %98.8 | 220,549 | 27.4 |
| gru | GRU mimari (LSTM yerine) | GRU | VAR | VAR | %96.47 | %100.0 | 167,301 | 33.9 |
| bilstm | BiLSTM (cift yonlu) | BiLSTM | VAR | VAR | %94.12 | %100.0 | 506,245 | 39.2 |
| no_aug | Augmentation kapali | LSTM | YOK | VAR | %95.29 | %100.0 | 220,549 | 23.3 |
| no_pose | Poz noktalari sifirli (sadece el) | LSTM | VAR | YOK | %90.59 | %98.8 | 220,549 | 34.3 |
