# GEditBench-v2 Subtask ELO Report: LongCat_Image_Edit

## Summary

| Metric | Value |
| --- | --- |
| Subtasks | 23 |
| Mean VC ELO | 1018.8 |
| Mean VQ ELO | 948.7 |
| Mean IF ELO | 919.0 |
| Mean Overall ELO | 966.3 |
| Rank-1 Subtasks | 0 |
| Top-3 Subtasks | 7 |

## Rank Distribution

| Rank | Count |
| --- | --- |
| 2 | 1 |
| 3 | 6 |
| 4 | 5 |
| 5 | 4 |
| 6 | 7 |

## LongCat Image Edit By Subtask

| Subtask | Samples | VC | VQ | IF | Overall | Rank | Best Model | Best Overall | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `background_change` | 40 | 1088 | 911 | 962 | 985 | 5 | `FLUX2_klein_9b` | 1063 | -78 |
| `camera_motion` | 60 | 977 | 824 | 919 | 914 | 6 | `GPT_Image_1p5` | 1124 | -210 |
| `character_reference` | 40 | 973 | 920 | 974 | 956 | 4 | `GPT_Image_1p5` | 1110 | -154 |
| `chart_editing` | 45 | 987 | 905 | 852 | 917 | 5 | `Nano_Banana_Pro` | 1221 | -304 |
| `color_alteration` | 40 | 1047 | 984 | 1036 | 1014 | 3 | `Nano_Banana_Pro` | 1090 | -76 |
| `enhancement` | 60 | 1053 | 770 | 850 | 910 | 6 | `GPT_Image_1p5` | 1101 | -191 |
| `hybrid` | 100 | 1009 | 962 | 1002 | 991 | 4 | `GPT_Image_1p5` | 1042 | -51 |
| `in_image_text_translation` | 60 | 1377 | 1202 | 401 | 1036 | 3 | `Nano_Banana_Pro` | 1235 | -199 |
| `line2image` | 40 | 933 | 835 | 944 | 919 | 6 | `Nano_Banana_Pro` | 1064 | -145 |
| `material_modification` | 40 | 1004 | 979 | 978 | 986 | 3 | `Nano_Banana_Pro` | 1075 | -89 |
| `motion_change` | 46 | 893 | 976 | 995 | 965 | 6 | `Nano_Banana_Pro` | 1045 | -80 |
| `object_reference` | 40 | 954 | 889 | 927 | 922 | 5 | `GPT_Image_1p5` | 1119 | -197 |
| `openset` | 98 | 979 | 925 | 816 | 912 | 6 | `GPT_Image_1p5` | 1088 | -176 |
| `portrait_beautification` | 62 | 997 | 1002 | 1009 | 998 | 3 | `Nano_Banana_Pro` | 1044 | -46 |
| `relation_change` | 41 | 1023 | 936 | 887 | 951 | 6 | `Nano_Banana_Pro` | 1092 | -141 |
| `size_adjustment` | 40 | 1029 | 971 | 862 | 959 | 6 | `GPT_Image_1p5` | 1076 | -117 |
| `style_reference` | 40 | 838 | 1008 | 835 | 906 | 5 | `Nano_Banana_Pro` | 1133 | -227 |
| `style_transfer` | 51 | 1047 | 945 | 957 | 983 | 4 | `GPT_Image_1p5` | 1035 | -52 |
| `subject_addition` | 40 | 972 | 1008 | 1001 | 998 | 4 | `Nano_Banana_Pro` | 1027 | -29 |
| `subject_removal` | 40 | 1086 | 919 | 1032 | 1005 | 3 | `Nano_Banana_Pro` | 1079 | -74 |
| `subject_replace` | 40 | 959 | 1029 | 1075 | 1025 | 2 | `FLUX2_klein_9b` | 1056 | -31 |
| `text_editing` | 87 | 1067 | 918 | 988 | 984 | 3 | `Nano_Banana_Pro` | 1122 | -138 |
| `tone_transfer` | 48 | 1141 | 1003 | 836 | 988 | 4 | `Nano_Banana_Pro` | 1080 | -92 |

## Sorted By LongCat Rank

| Rank | Subtask | Overall | VC | VQ | IF | Gap to Best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | `subject_replace` | 1025 | 959 | 1029 | 1075 | -31 |
| 3 | `in_image_text_translation` | 1036 | 1377 | 1202 | 401 | -199 |
| 3 | `color_alteration` | 1014 | 1047 | 984 | 1036 | -76 |
| 3 | `subject_removal` | 1005 | 1086 | 919 | 1032 | -74 |
| 3 | `portrait_beautification` | 998 | 997 | 1002 | 1009 | -46 |
| 3 | `material_modification` | 986 | 1004 | 979 | 978 | -89 |
| 3 | `text_editing` | 984 | 1067 | 918 | 988 | -138 |
| 4 | `subject_addition` | 998 | 972 | 1008 | 1001 | -29 |
| 4 | `hybrid` | 991 | 1009 | 962 | 1002 | -51 |
| 4 | `tone_transfer` | 988 | 1141 | 1003 | 836 | -92 |
| 4 | `style_transfer` | 983 | 1047 | 945 | 957 | -52 |
| 4 | `character_reference` | 956 | 973 | 920 | 974 | -154 |
| 5 | `background_change` | 985 | 1088 | 911 | 962 | -78 |
| 5 | `object_reference` | 922 | 954 | 889 | 927 | -197 |
| 5 | `chart_editing` | 917 | 987 | 905 | 852 | -304 |
| 5 | `style_reference` | 906 | 838 | 1008 | 835 | -227 |
| 6 | `motion_change` | 965 | 893 | 976 | 995 | -80 |
| 6 | `size_adjustment` | 959 | 1029 | 971 | 862 | -117 |
| 6 | `relation_change` | 951 | 1023 | 936 | 887 | -141 |
| 6 | `line2image` | 919 | 933 | 835 | 944 | -145 |
| 6 | `camera_motion` | 914 | 977 | 824 | 919 | -210 |
| 6 | `openset` | 912 | 979 | 925 | 816 | -176 |
| 6 | `enhancement` | 910 | 1053 | 770 | 850 | -191 |

## Column Meaning

- `VC`: Visual Consistency ELO.
- `VQ`: Visual Quality ELO.
- `IF`: Instruction Following ELO.
- `Overall`: joint ELO over VC, VQ, and IF.
- `Gap`: `FLUX2_klein_9b Overall ELO - Best Overall ELO`; `0` means this model ranks first on that subtask.

## Best-Performing Subtasks

None
