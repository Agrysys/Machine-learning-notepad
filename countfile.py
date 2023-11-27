import os

directories = [
    'dataset/tracehold/Test/Mentah',
    'dataset/tracehold/Test/Bukan',
    'dataset/tracehold/Test/Matang',
    'dataset/tracehold/Train/Bukan',
    'dataset/tracehold/Train/Matang',
    'dataset/tracehold/Train/Mentah'
]

for directory in directories:
    num_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    print(f'The directory {directory} contains {num_files} files.')
