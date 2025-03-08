import pynvml
import time
import subprocess
import sys

def initialize_nvml():
    """Initialize the NVML library."""
    try:
        pynvml.nvmlInit()
        print("NVML initialized successfully.")
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
        sys.exit(1)

def shutdown_nvml():
    """Shutdown the NVML library."""
    try:
        pynvml.nvmlShutdown()
        print("NVML shutdown successfully.")
    except pynvml.NVMLError as e:
        print(f"Failed to shutdown NVML: {e}")

def get_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    total = mem_info.total
    used = mem_info.used
    free = mem_info.free
    free_percent = (free / total) * 100

    print("Total GPU memory:", total)
    print("Used GPU memory:", used)
    print("Free GPU memory:", free)
    print("Free GPU memory (%):", free_percent)
    
    return free_percent

def run_when_gpu_free(script_path, min_free_percent):
    print(f"Monitoring GPU memory... Waiting for {min_free_percent}% free memory")
    
    while True:
        free_percent = get_gpu_memory_usage()
        print(f"Current free GPU memory: {free_percent:.2f}%")
        
        if free_percent >= min_free_percent:
            print(f"GPU memory requirement met ({free_percent:.2f}% free). Running script '{script_path}'...")
            try:
                process = subprocess.Popen(['python', script_path],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        bufsize=1)
                
                print("\nScript Output:")
                print("-" * 50)
                
                while True:
                    stdout_line = process.stdout.readline()
                    if stdout_line == '' and process.poll() is not None:
                        break
                    if stdout_line:
                        print(stdout_line.strip())
                
                stderr_output = process.stderr.read()
                if stderr_output:
                    print("\nScript Errors:")
                    print("-" * 50)
                    print(stderr_output.strip())
                
                return_code = process.wait()
                if return_code == 0:
                    print("\nScript completed successfully!")
                else:
                    print(f"\nScript failed with return code {return_code}")
                break
                
            except subprocess.SubprocessError as e:
                print(f"Error running script: {e}")
                break
            except FileNotFoundError:
                print(f"Script file '{script_path}' not found!")
                break
        else:
            print(f"Insufficient free memory ({free_percent:.2f}% < {min_free_percent}%). Waiting...")
            time.sleep(60) 

if __name__ == "__main__":
    # Initialize NVML before using any GPU-related functions
    initialize_nvml()

    try:
        if len(sys.argv) < 2:
            print("Please provide the script path as an argument")
            print("Usage: python this_script.py your_script.py")
            sys.exit(1)
        
        target_script = sys.argv[1]
        run_when_gpu_free(target_script, min_free_percent=55)
    
    finally:
        # Ensure NVML is properly shut down when the script exits
        shutdown_nvml()