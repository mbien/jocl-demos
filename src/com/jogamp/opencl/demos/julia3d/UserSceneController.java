package com.jogamp.opencl.demos.julia3d;

import com.jogamp.opencl.demos.julia3d.structs.RenderingConfig;
import com.jogamp.opencl.demos.julia3d.structs.Vec;
import java.awt.Component;
import java.awt.Point;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;

import static java.lang.Math.*;
import static com.jogamp.opencl.demos.julia3d.Renderer.*;

/**
 * Utility class for interacting with a scene. Supports rotation and zoom around origin.
 * @author Michael Bien
 */
public class UserSceneController {

    private Point dragstart;
    private RenderingConfig model;
    private Renderer view;

    private enum MOUSE_MODE { DRAG_ROTATE, DRAG_ZOOM }
    private MOUSE_MODE dragmode = MOUSE_MODE.DRAG_ROTATE;


    public void init(Renderer view, Component component, RenderingConfig model) {
        initMouseListeners(component);
        this.view = view;
        this.model = model;
    }

    private void initMouseListeners(Component component) {

        MouseAdapter mouseAdapter = new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {

                int x = e.getX();
                int y = e.getY();

                switch (dragmode) {
                    case DRAG_ROTATE:
                        if (dragstart != null) {
                            int height = model.getHeight();
                            int width = model.getWidth();

                            int ry = height - y - 1;
                            int baseMu1 = width - MU_RECT_SIZE - 2;
                            int baseMu2 = 1;
                            int baseMu3 = width - MU_RECT_SIZE - 2;
                            int baseMu4 = MU_RECT_SIZE + 2;

                            if ((x >= baseMu1 && x <= baseMu1 + MU_RECT_SIZE) && (ry >= baseMu2 && ry <= baseMu2 + MU_RECT_SIZE)) {
                                float[] mu = model.getMu();
                                mu[0] = 3.f * ( x - baseMu1) / (float)MU_RECT_SIZE - 1.5f;
                                mu[1] = 3.f * (ry - baseMu2) / (float)MU_RECT_SIZE - 1.5f;
                                model.setMu(mu);
                            } else if ((x >= baseMu3 && x <= baseMu3 + MU_RECT_SIZE) && (ry >= baseMu4 && ry <= baseMu4 + MU_RECT_SIZE)) {
                                float[] mu = model.getMu();
                                mu[2] = 3.f * ( x - baseMu3) / (float)MU_RECT_SIZE - 1.5f;
                                mu[3] = 3.f * (ry - baseMu4) / (float)MU_RECT_SIZE - 1.5f;
                                model.setMu(mu);
                            } else {
                                rotateCameraYbyOrig(0.01f * (x - dragstart.getX()));
                                rotateCameraXbyOrig(0.01f * (y - dragstart.getY()));
                            }
                        }
                        dragstart = e.getPoint();
                        view.update();
                        break;
                    case DRAG_ZOOM:
                        if (dragstart != null) {
                            float zoom = (float) ((y - dragstart.getY()) / 10.0f);
                            zoom(zoom);
                        }
                        dragstart = e.getPoint();
                        view.update();
                        break;
                }

            }

            @Override
            public void mousePressed(MouseEvent e) {
                switch (e.getButton()) {
                    case (MouseEvent.BUTTON1):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                    case (MouseEvent.BUTTON2):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                    case (MouseEvent.BUTTON3):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                switch (e.getButton()) {
                    case (MouseEvent.BUTTON1):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                    case (MouseEvent.BUTTON2):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                    case (MouseEvent.BUTTON3):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                }

                dragstart = null;
            }

            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                float zoom = e.getWheelRotation() * 0.1f;
                zoom(zoom);
                view.update();
            }

        };

        KeyAdapter keyAdapter = new KeyAdapter() {

            @Override
            public void keyPressed(KeyEvent e) {

                switch (e.getKeyChar()) {
                    case 'l':
                        model.setEnableShadow(model.getEnableShadow()==0 ? 1 : 0);
                        break;
                    case '1':
                        model.setEpsilon(max(0.0001f, model.getEpsilon() * 0.75f));
                        break;
                    case '2':
                        model.setEpsilon(model.getEpsilon() * 1.f / 0.75f);
                        break;
                    case '3':
                        model.setMaxIterations(max(1, model.getMaxIterations() -1));
                        break;
                    case '4':
                        model.setMaxIterations(min(12, model.getMaxIterations()+1));
                        break;
                    case '5':
                        model.setSuperSamplingSize(max(1, model.getSuperSamplingSize() -1));
                        break;
                    case '6':
                        model.setSuperSamplingSize(min(5, model.getSuperSamplingSize() +1));
                        break;
                    default:
                        break;
                }
                view.update();

            }

        };

        component.addKeyListener(keyAdapter);

        component.addMouseListener(mouseAdapter);
        component.addMouseMotionListener(mouseAdapter);
        component.addMouseWheelListener(mouseAdapter);

    }
    private void zoom(float zoom) {
        Vec orig = model.getCamera().getOrig();
        orig.setX(orig.getX()+zoom)
            .setY(orig.getY()+zoom)
            .setZ(orig.getZ()+zoom);
    }

    private void rotateLightX(float k) {
        float[] light = model.getLight();
        float y = light[1];
        float z = light[2];
        light[1] = (float) ( y * cos(k) + z * sin(k));
        light[2] = (float) (-y * sin(k) + z * cos(k));
        model.setLight(light);
    }

    private void rotateLightY(float k) {
        float[] light = model.getLight();
        float x = light[0];
        float z = light[2];
        light[0] = (float) (x * cos(k) - z * sin(k));
        light[2] = (float) (x * sin(k) + z * cos(k));
        model.setLight(light);
    }

    private void rotateCameraXbyOrig(double k) {
        Vec orig = model.getCamera().getOrig();
        float y = orig.getY();
        float z = orig.getZ();
        orig.setY((float) ( y * cos(k) + z * sin(k)));
        orig.setZ((float) (-y * sin(k) + z * cos(k)));
    }

    private void rotateCameraYbyOrig(double k) {
        Vec orig = model.getCamera().getOrig();
        float x = orig.getX();
        float z = orig.getZ();
        orig.setX((float) (x * cos(k) - z * sin(k)));
        orig.setZ((float) (x * sin(k) + z * cos(k)));
    }


    public final static void vadd(Vec v, Vec a, Vec b) {
        v.setX(a.getX() + b.getX());
        v.setY(a.getY() + b.getY());
        v.setZ(a.getZ() + b.getZ());
    }

    public final static void vsub(Vec v, Vec a, Vec b) {
        v.setX(a.getX() - b.getX());
        v.setY(a.getY() - b.getY());
        v.setZ(a.getZ() - b.getZ());
    }

    public final static void vmul(Vec v, float s, Vec b) {
        v.setX(s * b.getX());
        v.setY(s * b.getY());
        v.setZ(s * b.getZ());
    }

    public final static float vdot(Vec a, Vec b) {
        return a.getX() * b.getX()
             + a.getY() * b.getY()
             + a.getZ() * b.getZ();
    }

    public final static void vnorm(Vec v) {
        float s = (float) (1.0f / sqrt(vdot(v, v)));
        vmul(v, s, v);
    }

    public final static void vxcross(Vec v, Vec a, Vec b) {
        v.setX(a.getY() * b.getZ() - a.getZ() * b.getY());
        v.setY(a.getZ() * b.getX() - a.getX() * b.getZ());
        v.setZ(a.getX() * b.getY() - a.getY() * b.getX());
    }


}