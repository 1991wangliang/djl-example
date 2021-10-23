package com.codingapi.djl.example;

import org.junit.jupiter.api.Test;

/**
 * @author lorne
 * @since 1.0.0
 */
class YoLoV5Test {

    @Test
    void predict() throws Exception{
        YoLoV5 yolo = new YoLoV5();
        yolo.predict("images/img.png");
    }
}
