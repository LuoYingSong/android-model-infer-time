// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


package org.pytorch.demo.vit4mnist;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import com.alibaba.fastjson.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable {

    private Module mModule;

    private TextView mResultTextView;
    private Button mRecognizeButton;

    private static final float MNISI_STD = 0.1307f;
    private static final float MNISI_MEAN = 0.3081f;
    private static final float BLANK = - MNISI_STD / MNISI_MEAN;
    private static final float NON_BLANK = (1.0f - MNISI_STD) / MNISI_MEAN;
    private static final int MNIST_IMAGE_SIZE = 28;


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mResultTextView = findViewById(R.id.resultTextView);
//        mDrawView = findViewById(R.id.drawview);
        mRecognizeButton = findViewById(R.id.recognizeButton);
//        mClearButton = findViewById(R.id.clearButton);

        mRecognizeButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });


        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "MX_Res_opp_60_30_user0.ptl"));
//            mModule = Module.load(assetFilePath(this,"model.ptl"));
//            System.out.println();
        } catch (IOException e) {
            Log.e("VIT4MNIST", "Error reading assets", e);
            finish();
        }
    }

    public void run() {
        final int result = recognize();
        if (result == -1) return;
        runOnUiThread(() -> {
//            mResultTextView.clearComposingText();
            mResultTextView.setText(result + " MS");
        });
    }

    private int recognize() {
        // load data
        float[] inputs, mapRel, mapRec;
        try {
            inputs = loadData(MainActivity.assetFilePath(getApplicationContext(), "inputs.data"));
            mapRel = loadData(MainActivity.assetFilePath(getApplicationContext(), "send.data"));
            mapRec = loadData(MainActivity.assetFilePath(getApplicationContext(), "rec.data"));
        }catch (IOException e) {
            System.out.println("error");
            Log.e("VIT4MNIST", "Error reading assets", e);
            inputs = new float[1*5*60*9];
            mapRel = new float[5*10];
            mapRec = new float[5*10];
            finish();
        }
        long[] shape = new long[]{1,5,60,9};
        long[] mapShape = new long[]{10,5};

        Tensor inputTensor = Tensor.fromBlob(inputs,shape);
        Tensor recTensor = Tensor.fromBlob(mapRec,mapShape);
        Tensor sendTensor = Tensor.fromBlob(mapRel,mapShape);
        Long beginTime = System.currentTimeMillis();
        for(int i = 0; i < 100; i++) // infer 100 round
            mModule.forward(IValue.from(inputTensor),IValue.from(sendTensor),IValue.from(recTensor)).toTensor();
        Long endTime = System.currentTimeMillis();
        return (int) (endTime - beginTime);
    }

    private float[] loadData(String path) throws IOException {
        File file = new File(path);
        BufferedReader br = new BufferedReader(new FileReader(file));
        List<Float> valueList = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            valueList.add(Float.parseFloat(line.trim()));
        }
        float[] out = new float[valueList.size()];
        int idx = 0;
        for(final Float v : valueList){
            out[idx++] = v;
        }
        return out;
    }
}