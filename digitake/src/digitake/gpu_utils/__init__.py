from subprocess import check_output


def show_gpu():
  gpu_info = check_output(['nvidia-smi']).decode()
  #gpu_info = '\n'.join(gpu_info)
  if gpu_info.find('failed') >= 0:
    print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
  else:
    print(gpu_info)
