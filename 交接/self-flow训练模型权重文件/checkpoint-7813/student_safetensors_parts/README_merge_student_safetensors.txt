student.safetensors was split into 500MB parts because the upload connector has a 512MB single-file limit.

To restore the original file after downloading all parts into the same folder:

cat student.safetensors.part_* > student.safetensors

Original source path:
/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow/checkpoint-7813/student.safetensors
