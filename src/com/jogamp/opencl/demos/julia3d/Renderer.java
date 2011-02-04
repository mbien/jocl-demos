package com.jogamp.opencl.demos.julia3d;

import com.jogamp.opencl.demos.julia3d.structs.RenderingConfig;
import com.jogamp.opengl.util.awt.TextRenderer;
import java.awt.Canvas;
import java.awt.Font;
import java.nio.FloatBuffer;
import java.util.Timer;
import java.util.TimerTask;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;

import static com.jogamp.common.nio.Buffers.*;
import static javax.media.opengl.GL2.*;
import static java.lang.String.*;

/**
 * JOGL renderer for displaying the julia set.
 * @author Michael Bien
 */
public class Renderer implements GLEventListener {

    public final static int MU_RECT_SIZE = 80;

    private final Julia3d julia3d;
    private final GLCanvas canvas;
    private final RenderingConfig config;
    private final FloatBuffer juliaSlice;
    private final UserSceneController usi;
    private final TextRenderer textRenderer;

    private TimerTask task;
    private final Timer timer;

    public Renderer(final Julia3d julia3d) {
        
        this.julia3d = julia3d;
        this.config = julia3d.config;

        timer = new Timer(true);

        juliaSlice = newDirectFloatBuffer(MU_RECT_SIZE * MU_RECT_SIZE * 4);

        canvas = new GLCanvas(new GLCapabilities(GLProfile.get(GLProfile.GL2)));
        canvas.addGLEventListener(this);

        usi = new UserSceneController();
        usi.init(this, canvas, config);

        textRenderer = new TextRenderer(new Font("Helvetica", Font.BOLD, 14), true, true, null, false);

    }

    public void init(GLAutoDrawable drawable) {
        drawable.getGL().getGL2().glMatrixMode(GL_PROJECTION);
    }

    void update() {
        julia3d.update(false);
        canvas.display();
    }

    public void display(GLAutoDrawable drawable) {

        //compute
        julia3d.compute(config.getActvateFastRendering() == 1);

        GL2 gl = drawable.getGL().getGL2();
        gl.glClear(GL_COLOR_BUFFER_BIT);

        // draw julia set
	gl.glRasterPos2i(0, 0);
	gl.glDrawPixels(config.getWidth(), config.getHeight(), GL_RGB, GL_FLOAT, julia3d.getPixelBuffer());


        // Draw Mu constant
        int width = config.getWidth();
        int height = config.getHeight();
        float[] mu = config.getMu();

	gl.glEnable(GL_BLEND);
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            int baseMu1 = width - MU_RECT_SIZE - 2;
            int baseMu2 = 1;
            drawJuliaSlice(gl, baseMu1, baseMu2, mu[0], mu[1]);
            int baseMu3 = width - MU_RECT_SIZE - 2;
            int baseMu4 = MU_RECT_SIZE + 2;
            drawJuliaSlice(gl, baseMu3, baseMu4, mu[2], mu[3]);
	gl.glDisable(GL_BLEND);

	gl.glColor3f(1, 1, 1);
	int mu1 = (int) (baseMu1 + MU_RECT_SIZE * (mu[0] + 1.5f) / 3.f);
	int mu2 = (int) (baseMu2 + MU_RECT_SIZE * (mu[1] + 1.5f) / 3.f);
	gl.glBegin(GL_LINES);
            gl.glVertex2i(mu1 - 4, mu2);
            gl.glVertex2i(mu1 + 4, mu2);
            gl.glVertex2i(mu1, mu2 - 4);
            gl.glVertex2i(mu1, mu2 + 4);
	gl.glEnd();

	int mu3 = (int) (baseMu3 + MU_RECT_SIZE * (mu[2] + 1.5f) / 3.f);
	int mu4 = (int) (baseMu4 + MU_RECT_SIZE * (mu[3] + 1.5f) / 3.f);
	gl.glBegin(GL_LINES);
            gl.glVertex2i(mu3 - 4, mu4);
            gl.glVertex2i(mu3 + 4, mu4);
            gl.glVertex2i(mu3, mu4 - 4);
            gl.glVertex2i(mu3, mu4 + 4);
	gl.glEnd();

        // info text
        textRenderer.beginRendering(width, height);
        textRenderer.draw(format("Epsilon %.5f - Max. Iter. %d", config.getEpsilon(), config.getMaxIterations()), 8, 10);
        textRenderer.draw(format("Mu = (%.3f, %.3f, %.3f, %.3f)", mu[0], mu[1], mu[2], mu[3]), 8, 25);
        textRenderer.draw(format("Shadow %s - SuperSampling %dx%d - Fast rendering %s",
			config.getEnableShadow() == 1 ? "on" : "off",
                        config.getSuperSamplingSize(), config.getSuperSamplingSize(),
			config.getActvateFastRendering() == 1 ? "on" : "off"), 8, 40);
        textRenderer.endRendering();

        // timer task scheduling, delay gpu intensive high quality rendering
        if(task != null) {
            task.cancel();
        }
        if(config.getActvateFastRendering() == 1) {
            task = new TimerTask() {
                @Override
                public void run() {
                    config.setActvateFastRendering(0);
                    update();
                    config.setActvateFastRendering(1);
                }
            };
            timer.schedule(task, 2000);
        }
    }

    private void drawJuliaSlice(GL2 gl, int origX, int origY, float cR, float cI) {

        int index = 0;
        float invSize = 3.0f / MU_RECT_SIZE;
        for (int i = 0; i < MU_RECT_SIZE; ++i) {
            for (int j = 0; j < MU_RECT_SIZE; ++j) {

                float x = i * invSize - 1.5f;
                float y = j * invSize - 1.5f;

                int iter;
                for (iter = 0; iter < 64; ++iter) {
                    float x2 = x * x;
                    float y2 = y * y;
                    if (x2 + y2 > 4.0f) {
                        break;
                    }

                    float newx = x2 - y2 + cR;
                    float newy = 2.f * x * y + cI;
                    x = newx;
                    y = newy;
                }
                
                juliaSlice.put(index++, iter / 64.0f);
                juliaSlice.put(index++, 0.0f);
                juliaSlice.put(index++, 0.0f);
                juliaSlice.put(index++, 0.5f);
            }
        }

	gl.glRasterPos2i(origX, origY);
	gl.glDrawPixels(MU_RECT_SIZE, MU_RECT_SIZE, GL_RGBA, GL_FLOAT, juliaSlice);
    }


    public void reshape(GLAutoDrawable drawable, int x, int y, int newWidth, int newHeight) {

        config.setWidth(newWidth);
	config.setHeight(newHeight);

        GL2 gl = drawable.getGL().getGL2();

	gl.glViewport(0, 0, newWidth, newHeight);
	gl.glLoadIdentity();
	gl.glOrtho(-0.5f, newWidth - 0.5f, -0.5f, newHeight - 0.5f, -1.0f, 1.0f);

        julia3d.update(true);

    }

    public void dispose(GLAutoDrawable drawable) {
    }

    public Canvas getCanvas() {
        return canvas;
    }


}
