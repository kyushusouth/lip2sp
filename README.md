# プロジェクト作成手順

`pyenv local 3.11.4`でpythonのバージョンを指定

`poetry env info`で不必要な環境があれば`poetry remove`で消去

`poetry install`でパッケージのインストール

ルートに`.env`を作成し、以下の環境変数を定義
- `WANDB_API_KEY`: wandbのAPIキー