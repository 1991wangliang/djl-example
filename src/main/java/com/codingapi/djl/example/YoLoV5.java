package com.codingapi.djl.example;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * @author lorne
 * @since 1.0.0
 */
@Slf4j
public class YoLoV5 {

    private Predictor<Image, DetectedObjects> predictor;
    private final static int imageSize = 640;


    public YoLoV5() throws Exception {
        this.init();
    }

    public void init() throws Exception {
        //图片处理步骤
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize)); //调整尺寸
        pipeline.add(new ToTensor()); //处理为tensor类型
        //定义YoLov5的Translator
        Translator<Image, DetectedObjects> translator =  YoloV5Translator
                .builder()
                .setPipeline(pipeline)
                //labels信息定义
                .optSynset(Arrays.asList("HeartBreak", "octoberTreat", "PiercingSound", "baobao", "PeaceTreaty", "chuchangshunxu","binsizhuangtai","kapaijinyong","SurgarRush","Singlecombat","duimiankapai","CarrotHammer","kapajinyongSurgarRush","wofangshengyukapai","GasUnleash","IvoryStab","siwangzhuangtai","nengliangshumu","All-outshot","Surpriseinvasion","Insectivore","Endturn"))
                //预测的最小下限
                .optThreshold(0.3f)
                .build();

        //构建Model Criteria
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)//图片目标检测类型
                .optModelUrls("file:./models/best.torchscript.pt")//模型的路径
                .optTranslator(translator)//设置Translator
                .optProgress(new ProgressBar())//展示加载进度
                .build();
        //加载Model
        ZooModel<Image,DetectedObjects> model = ModelZoo.loadModel(criteria);

        //创建预测对象
        this.predictor = model.newPredictor();

    }


    public void predict(String file) throws Exception{
        //加载图片
        Image img = ImageFactory.getInstance().fromFile(Paths.get(file));
        //预测图片
        DetectedObjects results = predictor.predict(img);
        System.out.println(results);
        //创建用于保存的预测结果
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);
        //在预测结果上画出标记框
        ImageUtils.drawBoundingBoxes(imageSize,(BufferedImage) img.getWrappedImage(),results);
        //保存文件名称
        Path imagePath = outputDir.resolve("yolov5.png");
        // OpenJDK can"t save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        log.info("Detected objects image has been saved in: {}", imagePath);
    }
}
