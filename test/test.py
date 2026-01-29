import pandas as pd
import os
# import sys

# sys.path.append(os.path.abspath("../.."))

from demo.app import compare
import soundfile as sf


if __name__ == '__main__':
    
    correct = 0
    ground_truth = pd.read_csv('data/song_pairs.csv')
    songs_for_comparison = os.listdir('data/comparison')

    for song in songs_for_comparison:
        song_name = '.'.join(song.split('.')[:-1])
        # print(song_name)
        
        correct_song = ground_truth[ground_truth['comp_title'] == song_name]['ori_title'].values[0] \
            if song_name in ground_truth['comp_title'].values else None
        df_similar_songs = compare(f'data/comparison/{song}')
        top_songs = df_similar_songs[df_similar_songs['similarity'] == df_similar_songs['similarity'].max()]['item'].values
        top_songs = [top_song.split('\\')[-1] for top_song in top_songs]

        if f'{correct_song}.wav' in top_songs:
            correct += 1

        print(f'Song for comarison: {song}')
        print(f'Correct song: {correct_song}')
        print(f'Plagiarism candidates: {top_songs}')
        print('-'*50)

    print(f'Accuracy: {correct / len(songs_for_comparison)}')