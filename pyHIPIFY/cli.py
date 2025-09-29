import argparse
import os
import time
import multiprocessing as mp
from functools import partial
import shutil

from pyHIPIFY import hipify_python
from pyHIPIFY.hipify_python import is_hip_clang


# 将局部函数移到模块级别以解决pickle问题
def process_single_file(file_info, out_dir, proj_dir, show_progress, mode):
    fn, includes, ignores = file_info
    if includes and not any(fn.startswith(include) for include in includes):
        return None

    if any(fn.startswith(ignore) for ignore in ignores):
        return None

    file_path = os.path.join(proj_dir, fn)
    hipify_out_path = os.path.join(out_dir, fn[len(proj_dir)+1:] if len(proj_dir) > 0 else fn)
    
    # Ensure the output directory exists
    out_dir_path = os.path.dirname(hipify_out_path)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    
    # Process the file
    try:
        _, stats = hipify_python.process(
            file_path,
            hipify_out_path,
            show_progress=show_progress,
            hip_clang_launch=is_hip_clang())
        return stats
    except Exception as e:
        print("Error processing {}: {}".format(file_path, str(e)))
        return None


def main():
    parser = argparse.ArgumentParser(description='Top-level script for HIPifying, filling in most common parameters')
    parser.add_argument(
        '--out-of-place-only',
        action='store_true',
        help='Whether to only run hipify out-of-place on source files')

    parser.add_argument(
        '--project-directory',
        type=str,
        default='',
        help='The root of the project.',
        required=False)

    parser.add_argument(
        '--output-directory',
        type=str,
        default='',
        help='The Directory to Store the Hipified Project',
        required=False)

    parser.add_argument(
        '--include',
        type=str,
        default=[],
        nargs='+',
        help="The list of directories in caffe2 to hipify",
        required=False)

    parser.add_argument(
        '--ignores',
        type=str,
        default=[],
        nargs='+',
        help="The list of directories to ignore",
        required=False)

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress reporting')

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics')

    parser.add_argument(
        '--jobs',
        type=int,
        default=0,
        help='Number of parallel jobs to use for hipifying. Default is 0 (auto-detect).')

    parser.add_argument(
        '--file-processing-mode',
        type=str,
        default='all',
        choices=['all', 'convertible-only', 'shallow'],
        help='File processing mode: "all" (copy all files and convert CUDA files), '
             '"convertible-only" (only process convertible files), '
             '"shallow" (only process current directory, no subdirectories)')

    args = parser.parse_args()

    amd_build_dir = os.path.dirname(os.path.realpath(__file__))
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)))

    if args.project_directory:
        proj_dir = args.project_directory

    out_dir = proj_dir
    if args.output_directory:
        out_dir = args.output_directory

    show_progress = not args.no_progress

    # Collect overall stats
    overall_stats = {
        'total_files': 0,
        'total_lines': 0,
        'total_replacements': 0,
        'total_time': 0
    }

    start_time = time.time()

    # Use the includes from command line arguments
    includes = args.include
    
    # 添加调试信息
    if args.stats:
        print("Project directory: {}".format(proj_dir))
        print("Includes: {}".format(includes))
        print("Ignores: {}".format(args.ignores))
        print("File processing mode: {}".format(args.file_processing_mode))
    
    # 根据处理模式设置参数
    if args.file_processing_mode == 'shallow':
        # 对于浅层处理，我们需要修改includes参数以只包含当前目录
        if not includes:
            includes = ['*']
        files = hipify_python.find_files(proj_dir, includes, args.ignores, shallow=True, pytorch_caffe2_only=False)
    else:
        if not includes:
            includes = ['*']
        files = hipify_python.find_files(proj_dir, includes, ignores=args.ignores, pytorch_caffe2_only=False)
    
    # 添加调试信息
    if args.stats:
        print("Found {} files to process".format(len(files)))
        if len(files) <= 10:  # 只显示前10个文件，避免输出过多
            for f in files:
                print("  - {}".format(f))
        elif len(files) > 10:
            for f in files[:10]:
                print("  - {}".format(f))
            print("  ... and {} more files".format(len(files) - 10))

    if args.stats:
        print("\n" + "="*50)
        print("BEGIN HIPIFICATION")
        print("="*50)

    if args.jobs != 1:
        # Use parallel processing
        num_jobs = args.jobs if args.jobs > 0 else mp.cpu_count()
        
        # Prepare file info for processing
        file_infos = [(fn, args.include, args.ignores) for fn in files]
        
        # Create a partial function with fixed arguments
        process_func = partial(process_single_file, 
                              out_dir=out_dir, 
                              proj_dir=proj_dir, 
                              show_progress=show_progress,
                              mode=args.file_processing_mode)
        
        # Process files in parallel
        with mp.Pool(num_jobs) as pool:
            results = pool.map(process_func, file_infos)
        
        # Aggregate stats
        for result in results:
            if result is not None:
                overall_stats['total_files'] += result['files_processed']
                overall_stats['total_lines'] += result['lines_processed']
                overall_stats['total_replacements'] += result['replacements_made']
                overall_stats['total_time'] += result['elapsed_time']
    else:
        # Original sequential processing
        for fn in files:
            if args.include and not any(fn.startswith(include) for include in args.include):
                continue

            if any(fn.startswith(ignore) for ignore in args.ignores):
                continue

            is_out_of_place = args.out_of_place_only

            if is_out_of_place:
                file_path = os.path.join(proj_dir, fn)
                hipify_out_path = os.path.join(out_dir, fn[len(proj_dir)+1:] if len(proj_dir) > 0 else fn)
                
                # Ensure the output directory exists
                out_dir_path = os.path.dirname(hipify_out_path)
                if not os.path.exists(out_dir_path):
                    os.makedirs(out_dir_path)
                
                # Get stats if enabled
                if args.stats:
                    _, stats = hipify_python.process(
                        file_path,
                        hipify_out_path,
                        show_progress=show_progress,
                        hip_clang_launch=is_hip_clang())
                    overall_stats['total_files'] += stats['files_processed']
                    overall_stats['total_lines'] += stats['lines_processed']
                    overall_stats['total_replacements'] += stats['replacements_made']
                    overall_stats['total_time'] += stats['elapsed_time']

    # Show final stats
    if args.stats:
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print("CONVERSION STATISTICS")
        print("="*50)
        print("Total files processed: {}".format(overall_stats['total_files']))
        print("Total lines processed: {}".format(overall_stats['total_lines']))
        print("Total replacements made: {}".format(overall_stats['total_replacements']))
        print("Total processing time: {:.2f} seconds".format(overall_stats['total_time']))
        print("Wall clock time: {:.2f} seconds".format(elapsed_time))
        if overall_stats['total_time'] > 0:
            print("Processing efficiency: {:.2f} lines/second".format(
                overall_stats['total_lines'] / overall_stats['total_time']))
        print("="*50)

if __name__ == '__main__':
    main()