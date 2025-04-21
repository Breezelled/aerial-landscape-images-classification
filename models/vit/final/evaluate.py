import os
import torch
import numpy as np
from model import ViTForImageClassification
from dataset import get_data_loader

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, recall_score

def evaluate(y_test, scores, predicts, text=''):
    print('=========================================================')
    print(f'{text} performance:')
    accuracy = accuracy_score(y_test, predicts)
    acc1 = top_k_accuracy(y_test,scores,1)
    acc3 = top_k_accuracy(y_test,scores,3)
    acc5 = top_k_accuracy(y_test,scores,5)
    precision = precision_score(y_test, predicts, average='macro')
    recall = recall_score(y_test, predicts, average='macro')
    f1 = f1_score(y_test, predicts, average='macro')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Acc1:{acc1:.4f}')
    print(f'Acc3:{acc3:.4f}')
    print(f'Acc5:{acc5:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('---------Classification Report:---------')
    print(classification_report(y_test, predicts))
    print('=========================================================')

def top_k_accuracy(y_true, y_scores, k=1):
    top_k_preds = np.argsort(y_scores, axis=1)[:, -k:]
    correct = 0
    for true, topk in zip(y_true, top_k_preds):
        if true in topk:
            correct += 1
    return correct / len(y_true)

def draw_confusion_matrix(y_true, y_pred, save_dir, title=''):
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.title('Confusion Matrix {}'.format(title), fontsize=16)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_dir, f'{title}_Confusion.png'))

def get_attention(
        model_type='ViT',
        saved_path1='./save/ViT_all',
        saved_path2='./save/ViT_all',
        model1_name='Vit_all',
        model2_name='Vit_all',
        epoch1=10,
        epoch2=20,
        image_dir1="./Aerial_Landscapes",
        image_dir2="./Aerial_Landscapes",
        image_name1='Original',
        image_name2='Original',
        ex_name='freeze_all',
        batch_size=16,
        heatmap=True):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    classes = os.listdir(image_dir1)
    num_classes = len(classes)
    if model_type == 'ViT':
        model1 = ViTForImageClassification(num_classes=num_classes).to(device)
        checkpoint = torch.load(f"{saved_path1}/{epoch1}.pth", map_location=device)
        model1.load_state_dict(checkpoint['model_state_dict'])

        model2 = ViTForImageClassification(num_classes=num_classes).to(device)
        checkpoint = torch.load(f"{saved_path2}/{epoch2}.pth", map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])

        image_size = model1.image_size
        image_mean = model1.feature_extractor.image_mean
        image_std = model1.feature_extractor.image_std
    else:
        raise ValueError('model_type')

    # Load Dataset
    train_loader1, val_loader1, test_loader1 = get_data_loader(image_dir=image_dir1,
                                                            image_size=image_size,
                                                            expected_mean=image_mean,
                                                            expected_std=image_std,
                                                            batch_size=batch_size)
    
    train_loader2, val_loader2, test_loader2 = get_data_loader(image_dir=image_dir2,
                                                            image_size=image_size,
                                                            expected_mean=image_mean,
                                                            expected_std=image_std,
                                                            batch_size=batch_size)

    # Mkdir
    if heatmap == True:
        save_dir1 = os.path.dirname(f"./experiments/{ex_name}/{model1_name}_epoch{epoch1}/")
        save_dir2 = os.path.dirname(f"./experiments/{ex_name}/{model2_name}_epoch{epoch2}/")
        os.makedirs(save_dir1, exist_ok=True)
        os.makedirs(save_dir2, exist_ok=True)

    # Evaluate
    model1.eval()
    model2.eval()
    saved_img_dict1 = {}
    saved_img_dict2 = {}
    predicts1 = []
    predicts2 = []
    score_list1 = []
    score_list2 = []
    
    true_labels1 = []
    true_labels2 = []
    
    for i, ((images1, labels1, paths1), (images2, labels2, paths2)) in enumerate(zip(test_loader1,test_loader2)):
        # print(i, len(test_loader))

        with torch.no_grad():
            images1 = images1.to(device)
            images2 = images2.to(device)
            scores1, attentions1 = model1(images1)
            scores2, attentions2 = model2(images2)

            probs1 = torch.softmax(scores1, dim=1)
            preds_class1 = torch.argmax(probs1, dim=1)
            probs2 = torch.softmax(scores2, dim=1)
            preds_class2 = torch.argmax(probs2, dim=1)

            avg_attentions1 = attentions1[-1].mean(1)
            avg_attentions2 = attentions2[-1].mean(1)

            for pred_class1, pred_class2, label1, label2, avg_attention1, avg_attention2, image1, image2, path1,path2, score1, score2 in zip(preds_class1,
                                                                                                    preds_class2,
                                                                                                    labels1,
                                                                                                    labels2,
                                                                                                    avg_attentions1,
                                                                                                    avg_attentions2,
                                                                                                    images1, images2,
                                                                                                    paths1,paths2,
                                                                                                    scores1,scores2):
                true_labels1.append(label1.item())
                true_labels2.append(label2.item())
                predicts1.append(pred_class1.item())
                predicts2.append(pred_class2.item())

                score_list1.append(score1.cpu().numpy())
                score_list2.append(score2.cpu().numpy())
                
                if heatmap == False:
                    continue
                if label1 == pred_class1 and label2 == pred_class2 and f'{label1}_{pred_class1}' in saved_img_dict1 and f'{label2}_{pred_class2}' in saved_img_dict2:
                    continue
                cls_attention1 = avg_attention1[0, 1:]
                cls_attention2 = avg_attention2[0, 1:]

                # reshape
                heat_map1 = cls_attention1.reshape(14, 14).cpu().numpy()
                heat_map1 = (heat_map1 - heat_map1.min()) / (heat_map1.max() - heat_map1.min())
                heat_map1 = cv2.resize(heat_map1, (224, 224))
                heat_map2 = cls_attention2.reshape(14, 14).cpu().numpy()
                heat_map2 = (heat_map2 - heat_map2.min()) / (heat_map2.max() - heat_map2.min())
                heat_map2 = cv2.resize(heat_map2, (224, 224))

                # heatmap
                resized_img1 = image1.permute(1, 2, 0).cpu().numpy()
                resized_img1 = resized_img1 * image_std + image_mean
                resized_img1 = resized_img1.clip(0, 1)
                resized_img2 = image2.permute(1, 2, 0).cpu().numpy()
                resized_img2 = resized_img2 * image_std + image_mean
                resized_img2 = resized_img2.clip(0, 1)

                plt.imshow(resized_img1)
                plt.imshow(heat_map1, cmap='jet', alpha=0.3)
                plt.title(f"T:{classes[label1]} P:{classes[pred_class1]} Epoch:{epoch1} File:{path1}")
                plt.axis('off')
                plt.savefig(f'{save_dir1}/{path1}_{classes[label1]}_{classes[pred_class1]}.png')

                plt.imshow(resized_img2)
                plt.imshow(heat_map2, cmap='jet', alpha=0.3)
                plt.title(f"T:{classes[label2]} P:{classes[pred_class2]} Epoch:{epoch2} File:{path2}")
                plt.axis('off')
                plt.savefig(f'{save_dir2}/{path2}_{classes[label2]}_{classes[pred_class2]}.png')

                saved_img_dict1[f'{label1}_{pred_class1}'] = True
                saved_img_dict2[f'{label2}_{pred_class2}'] = True
                print(len(saved_img_dict1),len(saved_img_dict2))
                if len(saved_img_dict1)>20 and len(saved_img_dict2)>20:
                    heatmap=False

    evaluate(true_labels1, score_list1, predicts1, f'model:{model1_name}, image:{image_name1}, epoch:{epoch1}\n')
    draw_confusion_matrix(true_labels1, predicts1, save_dir=saved_path1, title=f"epoch{epoch1}_{image_name1}")
    
    evaluate(true_labels2, score_list2, predicts2, f'model:{model2_name}, image:{image_name2}, epoch:{epoch2}\n')
    draw_confusion_matrix(true_labels2, predicts2, save_dir=saved_path2, title=f"epoch{epoch2}_{image_name2}")


if __name__ == '__main__':
    epoch1 = 10
    epoch2 = 10    
    saved_path = './save/ViT_frozen_O'
    model_name = 'ViT_frozen_O'
    # get_attention(
    #     model_type='ViT',
    #     saved_path1=saved_path,
    #     saved_path2=saved_path,
    #     model1_name= model_name,
    #     model2_name= model_name,
    #     epoch1=epoch1,
    #     epoch2=epoch2,
    #     image_dir1="./Aerial_Landscapes",
    #     image_dir2="./Aerial_Landscapes_Adv",
    #     image_name1='Original',
    #     image_name2='Adversarial',
    #     ex_name='None',
    #     batch_size=16,
    #     heatmap=False)
    # get_attention(
    #     model_type='ViT',
    #     saved_path1=saved_path,
    #     saved_path2=saved_path,
    #     model1_name= model_name,
    #     model2_name= model_name,
    #     epoch1=epoch1,
    #     epoch2=epoch2,
    #     image_dir1="./Aerial_Landscapes_Gaussian",
    #     image_dir2="./Aerial_Landscapes_Salt",
    #     image_name1='Gaussian',
    #     image_name2='Salt',
    #     ex_name='None',
    #     batch_size=16,
    #     heatmap=False)

    
    saved_path1 = './save/ViT_frozen'
    model_name1 = 'ViT_frozen_O'
    saved_path2 = './save/ViT_all'
    model_name2 = 'ViT_all_O'
    get_attention(
        model_type='ViT',
        saved_path1=saved_path1,
        saved_path2=saved_path2,
        model1_name= model_name1,
        model2_name= model_name2,
        epoch1=epoch1,
        epoch2=epoch2,
        image_dir1="./Aerial_Landscapes",
        image_dir2="./Aerial_Landscapes",
        image_name1='Original',
        image_name2='Original',
        ex_name='None',
        batch_size=16,
        heatmap=False)