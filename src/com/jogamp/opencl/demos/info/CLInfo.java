/*
 * Created on Tuesday, September 07 2010 21:33
 */

package com.jogamp.opencl.demos.info;

import com.jogamp.common.JogampRuntimeException;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.util.ExceptionReporter;
import com.jogamp.opencl.util.JOCLVersion;
import java.awt.Container;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.UIManager;

/**
 * Displays OpenCL information in a table.
 * @author Michael Bien
 */
public class CLInfo {

    public static void main(String[] args) {
        
        Logger logger = Logger.getLogger(CLInfo.class.getName());

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ex) {
            logger.log(Level.INFO, null, ex);
        }

        logger.info("\n" + JOCLVersion.getAllVersions());

        try{
            CLPlatform.initialize();
        }catch(JogampRuntimeException ex) {
            logger.log(Level.SEVERE, null, ex);
            ExceptionReporter.appear("I tried hard but I really can't initialize JOCL. Is OpenCL properly set up?", ex);
            return;
        }

        JFrame frame = new JFrame("OpenCL Info");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        Container contentPane = frame.getContentPane();

        JEditorPane area = new JEditorPane();
        area.setContentType("text/html");
        area.setEditable(false);

        contentPane.add(new JScrollPane(area));

        String html = createOpenCLInfoHTML();

        area.setText(html.toString());

        frame.setSize(800, 600);
        frame.setVisible(true);

    }

    private static String createOpenCLInfoHTML() {

        StringBuilder html = new StringBuilder();

        html.append("<table border=\"1\">");
        CLPlatform[] platforms = CLPlatform.listCLPlatforms();

        // platforms
        List<Map<String, String>> platProps = new ArrayList<Map<String, String>>();
        List<Integer> spans = new ArrayList<Integer>();
        for (CLPlatform p : platforms) {
            platProps.add(p.getProperties());
            spans.add(p.listCLDevices().length);
        }
        fillTable(platProps, spans, html);

        // devices
        ArrayList<Map<String, String>> devProps = new ArrayList<Map<String, String>>();
        for (CLPlatform p : platforms) {
            CLDevice[] devices = p.listCLDevices();
            for (CLDevice d : devices) {
                devProps.add(d.getProperties());
            }
        }
        fillTable(devProps, html);
        html.append("</table>");

        return html.toString();
    }

    private static void fillTable(List<Map<String, String>> properties, StringBuilder sb) {
        ArrayList<Integer> spans = new ArrayList<Integer>(properties.size());
        for (int i = 0; i < properties.size(); i++) {
            spans.add(1);
        }
        fillTable(properties, spans, sb);
    }

    private static void fillTable(List<Map<String, String>> properties, List<Integer> spans, StringBuilder sb) {
        boolean header = true;
        for (String key : properties.get(0).keySet()) {
            sb.append("<tr>");
                cell(sb, key);
                int i = 0;
                for (Map<String, String> map : properties) {
                    cell(sb, spans.get(i), map.get(key), header);
                    i++;
                }
            sb.append("</tr>");
            header = false;
        }
    }

    private static void cell(StringBuilder sb, String value) {
        sb.append("<td>").append(value).append("</td>");
    }

    private static void cell(StringBuilder sb, int span, String value, boolean header) {
        if(header) {
            sb.append("<th colspan=\"").append(span).append("\">").append(value).append("</th>");
        }else{
            sb.append("<td colspan=\"").append(span).append("\">").append(value).append("</td>");
        }
    }
}
