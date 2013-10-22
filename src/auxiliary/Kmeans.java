package auxiliary;

import java.util.HashSet;
import java.util.Random;

/**
 *
 * @author MF1333020 孔繁宇
 */
public class Kmeans {

    public Kmeans() {
    }

    /*
     * Input double[numIns][numAtt] features, int K
     * Output double[K][numAtt] clusterCenters, int[numIns] clusterIndex
     * 
     * clusterCenters[k] should store the kth cluster center
     * clusterIndex[i] should store the cluster index which the ith sample belongs to
     */
    public void train(double[][] features, int K, double[][] clusterCenters, int[] clusterIndex) {
        int numIns = features.length;
        if (numIns == 0) return;
        int numAtt = features[0].length;
        
        //随机初始化中心点
        Random random = new Random();
        HashSet<Integer> checker = new HashSet<Integer>();
        for (int i = 0; i < K; ++i) {
            int center = 0;
            do {
                center = random.nextInt(numIns);
            } while (checker.contains(center));
            checker.add(center);
            
            clusterCenters[i] = features[center].clone();
        }
        
        //迭代更新
        boolean flag;
        while (true) {
            flag = false;
            
            //记录数组，辅助计算新中心点
            double[][] temp = new double[K][numAtt];
            int[] counter = new int[K];
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < numAtt; ++j) {
                    temp[i][j] = 0;
                }
                counter[i] = 0;
            }
            
            for (int i = 0; i < numIns; ++i) {
                int index = closestCluster(features[i], clusterCenters, K);
                
                //记录辅助值
                counter[index]++;
                for (int j = 0; j < numAtt; ++j) {
                    temp[index][j] += features[i][j];
                }
                
                if (index != clusterIndex[i]) {
                    //更新簇标号
                    flag = true;
                    clusterIndex[i] = index;
                }
            }
            
            if (flag) {
                //用辅助值更新蔟均值
                for (int i = 0; i < K; ++i) {
                    for (int j = 0; j < numAtt; ++j) {
                        clusterCenters[i][j] = temp[i][j] / counter[i];
                    }
                }
            } else {
                //已稳定，迭代结束
                break;
            };
        }
        
        return;
    }
    
    private int closestCluster(double[] feature, double[][] clusterCenters, int K) {
        int cluster = -1;
        double min = -1;
        
        for (int i = 0; i < K; ++i) {
            double dist = distance(feature, clusterCenters[i]);
            if (min < 0 || dist < min) {
                cluster = i;
                min = dist;
            }
        }
        return cluster;
    }
    
    //计算两个样本间的欧式距离
    private double distance(double[] a, double[] b) {
        if (a.length != b.length) return 0;
        
        int length = a.length;
        if (length == 0) return 0;
        
        double result = 0;
        for (int i = 0; i < length; ++i) {
            result += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return result;
    }
}
