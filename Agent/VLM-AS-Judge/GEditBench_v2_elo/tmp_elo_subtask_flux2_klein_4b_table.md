# GEditBench-v2 Subtask ELO Report: FLUX2_klein_4b

## Summary

| Metric | Value |
| --- | --- |
| Subtasks | 23 |
| Mean VC ELO | 1085.1 |
| Mean VQ ELO | 883.6 |
| Mean IF ELO | 895.6 |
| Mean Overall ELO | 957.8 |
| Rank-1 Subtasks | 0 |
| Top-3 Subtasks | 1 |

## Rank Distribution

| Rank | Count |
| --- | --- |
| 2 | 1 |
| 4 | 3 |
| 5 | 6 |
| 6 | 4 |
| 7 | 9 |

## FLUX2 klein 4b By Subtask

| Subtask | Samples | VC | VQ | IF | Overall | Rank | Best Model | Best Overall | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `background_change` | 40 | 1195 | 897 | 1025 | 1030 | 2 | `FLUX2_klein_9b` | 1052 | -22 |
| `camera_motion` | 60 | 1006 | 984 | 980 | 992 | 4 | `GPT_Image_1p5` | 1120 | -128 |
| `character_reference` | 40 | 926 | 912 | 1050 | 965 | 4 | `GPT_Image_1p5` | 1117 | -152 |
| `chart_editing` | 45 | 1090 | 924 | 889 | 970 | 4 | `Nano_Banana_Pro` | 1225 | -255 |
| `color_alteration` | 40 | 991 | 814 | 845 | 886 | 7 | `Nano_Banana_Pro` | 1111 | -225 |
| `enhancement` | 60 | 1055 | 845 | 921 | 952 | 5 | `GPT_Image_1p5` | 1100 | -148 |
| `hybrid` | 100 | 1066 | 841 | 900 | 941 | 7 | `Nano_Banana_Pro` | 1051 | -110 |
| `in_image_text_translation` | 60 | 1118 | 815 | 724 | 925 | 5 | `Nano_Banana_Pro` | 1250 | -325 |
| `line2image` | 40 | 821 | 1007 | 1126 | 993 | 5 | `Nano_Banana_Pro` | 1068 | -75 |
| `material_modification` | 40 | 1112 | 882 | 903 | 964 | 7 | `Nano_Banana_Pro` | 1079 | -115 |
| `motion_change` | 46 | 1162 | 837 | 884 | 957 | 7 | `Nano_Banana_Pro` | 1054 | -97 |
| `object_reference` | 40 | 1060 | 889 | 894 | 948 | 5 | `GPT_Image_1p5` | 1119 | -171 |
| `openset` | 100 | 1028 | 907 | 930 | 957 | 5 | `GPT_Image_1p5` | 1093 | -136 |
| `portrait_beautification` | 62 | 1167 | 881 | 900 | 971 | 6 | `Nano_Banana_Pro` | 1051 | -80 |
| `relation_change` | 41 | 1187 | 931 | 695 | 957 | 7 | `Nano_Banana_Pro` | 1092 | -135 |
| `size_adjustment` | 40 | 1237 | 925 | 699 | 967 | 6 | `GPT_Image_1p5` | 1084 | -117 |
| `style_reference` | 40 | 1006 | 950 | 865 | 945 | 5 | `GPT_Image_1p5` | 1141 | -196 |
| `style_transfer` | 51 | 1037 | 913 | 960 | 971 | 6 | `Seedream4p5` | 1041 | -70 |
| `subject_addition` | 40 | 1209 | 824 | 825 | 948 | 7 | `Nano_Banana_Pro` | 1031 | -83 |
| `subject_removal` | 40 | 1164 | 761 | 969 | 962 | 6 | `Nano_Banana_Pro` | 1089 | -127 |
| `subject_replace` | 40 | 1194 | 833 | 877 | 958 | 7 | `FLUX2_klein_9b` | 1055 | -97 |
| `text_editing` | 87 | 1049 | 863 | 795 | 905 | 7 | `Nano_Banana_Pro` | 1136 | -231 |
| `tone_transfer` | 48 | 1077 | 888 | 943 | 965 | 7 | `Nano_Banana_Pro` | 1085 | -120 |

## Sorted By FLUX Rank

| Rank | Subtask | Overall | VC | VQ | IF | Gap to Best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | `background_change` | 1030 | 1195 | 897 | 1025 | -22 |
| 4 | `camera_motion` | 992 | 1006 | 984 | 980 | -128 |
| 4 | `chart_editing` | 970 | 1090 | 924 | 889 | -255 |
| 4 | `character_reference` | 965 | 926 | 912 | 1050 | -152 |
| 5 | `line2image` | 993 | 821 | 1007 | 1126 | -75 |
| 5 | `openset` | 957 | 1028 | 907 | 930 | -136 |
| 5 | `enhancement` | 952 | 1055 | 845 | 921 | -148 |
| 5 | `object_reference` | 948 | 1060 | 889 | 894 | -171 |
| 5 | `style_reference` | 945 | 1006 | 950 | 865 | -196 |
| 5 | `in_image_text_translation` | 925 | 1118 | 815 | 724 | -325 |
| 6 | `portrait_beautification` | 971 | 1167 | 881 | 900 | -80 |
| 6 | `style_transfer` | 971 | 1037 | 913 | 960 | -70 |
| 6 | `size_adjustment` | 967 | 1237 | 925 | 699 | -117 |
| 6 | `subject_removal` | 962 | 1164 | 761 | 969 | -127 |
| 7 | `tone_transfer` | 965 | 1077 | 888 | 943 | -120 |
| 7 | `material_modification` | 964 | 1112 | 882 | 903 | -115 |
| 7 | `subject_replace` | 958 | 1194 | 833 | 877 | -97 |
| 7 | `motion_change` | 957 | 1162 | 837 | 884 | -97 |
| 7 | `relation_change` | 957 | 1187 | 931 | 695 | -135 |
| 7 | `subject_addition` | 948 | 1209 | 824 | 825 | -83 |
| 7 | `hybrid` | 941 | 1066 | 841 | 900 | -110 |
| 7 | `text_editing` | 905 | 1049 | 863 | 795 | -231 |
| 7 | `color_alteration` | 886 | 991 | 814 | 845 | -225 |

## Column Meaning

- `VC`: Visual Consistency ELO.
- `VQ`: Visual Quality ELO.
- `IF`: Instruction Following ELO.
- `Overall`: joint ELO over VC, VQ, and IF.
- `Gap`: `FLUX2_klein_9b Overall ELO - Best Overall ELO`; `0` means this model ranks first on that subtask.

## Best-Performing Subtasks

None
