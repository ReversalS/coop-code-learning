# do the filtering and statistics work here
from tqdm import tqdm


def filter_paths(dataset):
    total_paths = 0
    total_length = 0
    context_count = 0
    max_path_length = -1
    filtered_dataset = []
    for path_contexts in tqdm(dataset):
        filtered_context = []
        context_path_count = 0
        fragment = ''   # splitting '\n' may break down paths, but it can be recontructed
        for line in path_contexts.split(' '):
            if not line:
                continue
            if fragment:
                line = fragment + '<\s>' + line
                fragment = ''
            try:
                left_token, path, right_token = line.split(',')[:3]
                if line[0] == '[':  # discard TypeDecl
                    continue
                total_length += len(line.split('-'))
                total_paths += 1
                filtered_context.append(line)
            except:
                fragment += line
        context_count += 1
        filtered_dataset.append(filtered_context)
    print(context_count)
    print(f'Average path count: {total_paths / context_count}; average path length: {total_length / total_paths}')
    return filtered_dataset
    