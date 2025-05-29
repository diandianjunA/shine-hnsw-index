import psutil
import time


# script that checks whether vmtouch is done with mapping
# must be executed directly on the server that runs vmtouch

def get_vmtouch_pid():
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            if proc.info["cmdline"] and "vmtouch" in proc.info["cmdline"][0]:
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return None


def mapping_done(process: psutil.Process):
    mem_usage = process.memory_info()
    return mem_usage.rss >= mem_usage.vms - 100 * 1024 ** 2  # leave 100 MB headroom


if __name__ == "__main__":
    pid = get_vmtouch_pid()
    p = psutil.Process(pid)

    while True:
        if not p.is_running():
            print("vmtouch process is no longer running.")
            break

        if mapping_done(p):
            print("vmtouch has likely switched into daemon mode")
            break

        # sleep briefly before the next check.
        time.sleep(1)
