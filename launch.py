import os
from modules import launch_utils

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    if args.test_server:
        configure_for_tests()

    start()


def mysql_heartbeat():
    """定时向MySQL写入数据, 实现心跳服务"""
    os.system("pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql")

    from aigcapi.mysql.heartbeat import Heartbeat
    heartbeat = Heartbeat(
        host=args.heartbeat_host, 
        user=args.heartbeat_user, 
        password=args.heartbeat_password, 
        database=args.heartbeat_database)

    heartbeat.listen(
        table=args.heartbeat_table,
        frequency=float(args.heartbeat_frequency),
        restart_time=10.0, 
        block=False)


def mount_distributed():
    """挂载共享目录"""
    cmd = "mkdir -p /base && mount -t nfs {} /base".format(args.nfs_model_base_dir)
    os.system(cmd)
    cmd = "mkdir -p /lora && mount -t nfs {} /lora".format(args.nfs_model_lora_dir)
    os.system(cmd)


if __name__ == "__main__":
    mount_distributed()
    mysql_heartbeat()
    main()