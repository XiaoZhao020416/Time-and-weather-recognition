def main(config):
    cls_predictor = ClsPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    batch_imgs = []
    batch_names = []
    cnt = 0

    # 保存到文件
    f=open('/home/aistudio/result.txt', 'w')

    for idx, img_path in enumerate(image_list):
        img = CV2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                         ) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            batch_results = cls_predictor.predict(batch_imgs)
            for number, result_dict in enumerate(batch_results):
                filename = batch_names[number]
                clas_ids = result_dict["class_ids"]
                scores_str = "[{}]".format(", ".join("{:.2f}".format(
                    r) for r in result_dict["scores"]))
                label_names = result_dict["label_names"]
                f.write("{} {}\n".format(filename, clas_ids[0]))
                print("{}:\tclass id(s): {}, score(s): {}, label_name(s): {}".
                      format(filename, clas_ids, scores_str, label_names))
            batch_imgs = []
            batch_names = []


    if cls_predictor.benchmark:
        cls_predictor.auto_logger.report()
    return