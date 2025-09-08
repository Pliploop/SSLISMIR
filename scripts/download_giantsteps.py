import mirdata
import os
import pandas as pd


DATA_HOME = 'data/giantsteps'


giantsteps = mirdata.initialize('giantsteps_key', data_home = DATA_HOME)
giantsteps.download()

## create a json metadata file joined on file name

key_folder = os.path.join(DATA_HOME, 'keys_gs+')
audio_folder = os.path.join(DATA_HOME, 'audio')

key_files = os.listdir(key_folder)
key_files = [f for f in key_files if f.endswith('.txt')]
audio_files = os.listdir(audio_folder)
audio_files = [f for f in audio_files if f.endswith('.mp3')]


metadata = []
for key_file in key_files:
    with open(os.path.join(key_folder, key_file), 'r') as f:
        key = f.read().strip()
    audio_file = os.path.join(audio_folder, key_file.replace('.txt', '.mp3'))
    metadata.append({
        'key': key,
        'file_path': audio_file
    })

## key to numerical label with pandas
df = pd.DataFrame(metadata)

df['label_id'] = df.key.astype('category').cat.codes
df.to_json(os.path.join(DATA_HOME, 'metadata.json'), orient='records')