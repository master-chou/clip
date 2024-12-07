
import json

# with open('/home/aiops/wangzh/data/RGBD-benchmark/annotations.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Counting/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Relative_Depth/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Spatial_Relation/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-13b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Visual_Correspondence/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-13b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Multi-view_Reasoning/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Object_Localization/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-13b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/blink/BLINK/Jigsaw/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/CV-Bench/test3d-depth.jsonl', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/CV-Bench/test-count.jsonl', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:

#
# with open('/home/aiops/wangzh/data/blink/BLINK/Spatial_Relation/output.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/all.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/all.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/gpt4o-outdoor.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor-new/all.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/gemini-indoor.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor-new/all.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/gemini-indoor.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/realworldqa/updated.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:

# with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/object_orientation.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_depth.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_size.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_spatial_position.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor/none_spatial.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor/orientation.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor/relative_depth.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor/relative_size.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:
# with open('/home/aiops/wangzh/data/scanner/indoor/spatial_relation.json', 'r') as reader1, open('/home/aiops/wangzh/llava-spat/llava-v1.5-7b-final-neg-lora_answers.txt', 'r') as reader2:

    reader1 = json.load(reader1)

    correct = 0
    total = 0
    for index, (line1, line2) in enumerate(zip(reader1, reader2), 1):
        total += 1

        answer = line2.strip()
        ground_truth = line1['answer']
        # ground_truth = json.loads(line1.strip())['answer']
        # length = len(ground_truth)
        flag = False
        # choices = json.loads(line1.strip())['choices']

        # if ground_truth in answer:
        #     correct += 1
        # import pdb;pdb.set_trace()
        # if ('A' in answer) and ('B' in answer) and ('C' in answer) and ('D' in answer):
        #     print('missed',index)
        #     continue
        # if ground_truth == '(A)':
        #     if 'left' in answer:
        #         correct += 1
        #         print("yes",index)
        # elif ground_truth == '(B)':
        #     if 'right' in answer:
        #         correct += 1 
        #         print("yes",index)
        # if ground_truth == '(A)':
        #     if 'second' in answer:
        #         correct += 1
        #         print("yes",index)
        # elif ground_truth == '(B)':
        #     if 'third' in answer:
        #         correct += 1 
        #         print("yes",index)
        # count_sed = answer.count('sed')
        # count_tird = answer.count('tird')
        if ground_truth == 0:
            if ('A' in answer) :
                correct += 1
                print("yes",index)

        elif ground_truth == 1:
            if ('B' in answer) :
                correct += 1 
                print("yes",index)

        
        elif ground_truth == 2:
            if ('C' in answer) :
                correct += 1 
                print("yes",index)

        
        elif ground_truth == 3:
            if ('D' in answer) :
                correct += 1 
                print("yes",index)
        # if ground_truth == answer:
        #     correct += 1 
        #     print("yes",index)
        
        # if ground_truth == '(A)':
        #     if 'A' in answer :
        #         correct += 1
        #         print("yes",index)
        # elif ground_truth == '(B)':
        #     if 'B' in answer:
        #         correct += 1 
        #         print("yes",index)
        # elif ground_truth == '(C)':
        #     if 'C' in answer:
        #         correct += 1 
        #         print("yes",index)
        # elif ground_truth == '(D)':
        #     if 'D' in answer:
        #         correct += 1 
        #         print("yes",index)
        # if ground_truth == '(A)':
        #     if choices[2] in answer:
        #         correct += 1
        #         print("yes",index)
        # elif ground_truth == '(B)':
        #     if choices[6] in answer:
        #         correct += 1 
        #         print("yes",index)
        # elif ground_truth == '(C)':
        #     if choices[10] in answer:
        #         correct += 1 
        #         print("yes",index)
        # elif ground_truth == '(D)':
        #     if choices[14] in answer:
        #         correct += 1 
        #         print("yes",index)
        
    print("correct =", correct)
    print("total =", total)
    print("acc =",correct/total)



    # correct = 0
    # total = 0
    # for index, (line1, line2) in enumerate(zip(reader1, reader2), 1):
    #     total += 1
    #     answer = line2.strip()
    #     ground_truth = json.loads(line1.strip())['answer'].strip(".")[0]
    #     length = len(ground_truth)
    #     flag = False
    #     import pdb;pdb.set_trace()
    #     if length == 1 and ground_truth.isalpha():
    #         flag = True
    #         answer = answer.split(".")[0]
    #     elif length == 2 or length == 3:
    #         flag = True
    #         answer = answer.split(",")[0]

    #     if flag:
    #         if answer.lower() == ground_truth.lower():
    #             correct += 1
    #     else:
    #         print("->", index)
    # print("correct =", correct)
    # print("total =", total)