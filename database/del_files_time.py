
import os
import time
import shutil
# ファイルの保持時間（秒）
FILE_LIFETIME = 600  #：sec
UPLOAD_FOLDER = 'app/uploads'
fig_dir = "app/static/database_figures/advanced_results"
def delete_old_files(fol):
    while True:
        now = time.time()
        for filename in os.listdir(fol):
            file_path = os.path.join(fol, filename)
            if os.path.isdir(file_path):
                # ファイルの作成時間を取得し、現在の時刻と比較
                file_creation_time = os.path.getctime(file_path)
                print(file_creation_time)
                if now - file_creation_time > FILE_LIFETIME:
                    shutil.rmtree(file_path)
                    # print(f"Deleted: {file_path}")
         # 1分ごとにチェック

if __name__ == "__main__":
    delete_old_files(UPLOAD_FOLDER)
    delete_old_files(fig_dir)