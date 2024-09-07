/* This file is copyright (c) 2024 Srikumar Krishnamoorthy
 * 
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.List;
import java.util.ArrayList;

public class KBinsDiscretizer {
    final int nBins;
    private float[] binEdges;
	boolean missingValueFound = false;

    public KBinsDiscretizer(int nBins) {
        this.nBins = nBins;
    }

	private float getMin(int[] A){
		int minValue = Integer.MAX_VALUE;
		for (int value : A){
			if (value==-999) {missingValueFound=true;continue;}//ignore missing value
            if (value < minValue) {
                minValue = value;
            }
		}
		return minValue;
    }

    private float getMin(float[] A){
        float minValue = Float.MAX_VALUE;
        for (float value : A) {
			if (value==-999) {missingValueFound=true;continue;}//ignore missing value
            if (value < minValue) {
                minValue = value;
            }
        }
		return minValue;
    }

    private float getMax(int[] A){
		int maxValue = Integer.MIN_VALUE;
		for (int value : A){
			if (value==-999) {missingValueFound=true;continue;}//ignore missing value
            if (value > maxValue) {
                maxValue = value;
            }
		}
		return maxValue;
    }

    private float getMax(float[] A){
        float maxValue = Float.MIN_VALUE;
        for (float value : A) {
			if (value==-999) {missingValueFound=true;continue;}//ignore missing value 
            if (value > maxValue) {
                maxValue = value;
            }
        }
		return maxValue;
    }

    public void fit(float[] column) {
        float colMin = getMin(column);
        float colMax = getMax(column);    
        if (colMin == colMax) {
            binEdges = new float[]{-Float.MAX_VALUE, Float.MAX_VALUE};
        } else {
			if (missingValueFound){
				List<Float> tmp = new ArrayList<Float>();
				for (Float f : column) if (f!=-999) tmp.add(f);
				float[] columnValid = new float [tmp.size()];
				for (int i=0;i<tmp.size();i++) columnValid[i] = tmp.get(i);
				binEdges = calculateQuantileEdges(columnValid, nBins);//passing only the non-missing entries
			}else{
				binEdges = calculateQuantileEdges(column, nBins);
			}
            binEdges = adjustBinEdges(binEdges, nBins);
        }
    }

    public void fit(int[] column) {
        float colMin = getMin(column);
        float colMax = getMax(column);
        if (colMin == colMax) {
            binEdges = new float[]{-Float.MAX_VALUE, Float.MAX_VALUE};
        } else {
			if (missingValueFound){
				List<Integer> tmp = new ArrayList<Integer>();
				for (Integer f : column) if (f!=-999) tmp.add(f);
				int[] columnValid = new int [tmp.size()];
				for (int i=0;i<tmp.size();i++) columnValid[i] = tmp.get(i);
				binEdges = calculateQuantileEdges(columnValid, nBins);//passing only the non-missing entries
			}else{
				binEdges = calculateQuantileEdges(column, nBins);
				binEdges = adjustBinEdges(binEdges, nBins);
			}
        }
    }

    private float[] adjustBinEdges(float[] binEdges, int nBins) {
		float[] adjustedEdges = new float[binEdges.length];
		int index = 0;
		float lastEdge = Float.NaN;
		for (float edge : binEdges) {
			if ((index == 0 || edge != lastEdge) && !(index > 0 && edge == 0.0f && lastEdge == 0.0f)) {
				adjustedEdges[index++] = edge;
				lastEdge = edge;
			}
		}
		return Arrays.copyOfRange(adjustedEdges, 0, index);
	}

    private float[] calculateQuantileEdges(float[] column, int nBins) {
        Float[] quantilesTmp = IntStream.range(0, nBins + 1)
            .mapToDouble(q -> q * 100.0 / nBins) 
			.mapToObj(d -> (float) d)           // Convert double to float
            .toArray(Float[]::new);               // Collect to float[]
        
		float[] quantiles = new float[quantilesTmp.length];
        for (int i = 0; i < quantilesTmp.length; i++) {
            quantiles[i] = quantilesTmp[i];
        }

        float[] binEdges = new float[quantiles.length];
        for (int i = 0; i < quantiles.length; i++) {
            binEdges[i] = calculatePercentile(column, quantiles[i]); 
        }
		binEdges = adjustBinEdges(binEdges, nBins);
        return binEdges;
    }

    private float[] calculateQuantileEdges(int[] column, int nBins) {
        double[] quantilesTmp = IntStream.range(0, nBins + 1)
											.mapToDouble(q -> q * 100.0 / nBins)  // Convert to double
											.toArray();                           // Convert to double[] array

		float[] quantiles = new float[quantilesTmp.length];
        for (int i = 0; i < quantilesTmp.length; i++) {
            quantiles[i] = (float) quantilesTmp[i];
        }

        float[] binEdges = new float[quantiles.length];
        for (int i = 0; i < quantiles.length; i++) {
            binEdges[i] = calculatePercentile(column, quantiles[i]); 
        }
		
		binEdges = adjustBinEdges(binEdges, nBins);
		return binEdges;
    }

    private float calculatePercentile(float[] column, double percentile) {
        float[] sortedColumn = Arrays.copyOf(column, column.length); 
		Arrays.sort(sortedColumn); 
        int index = (int) Math.ceil(percentile / 100.0 * (sortedColumn.length - 1));
        return sortedColumn[Math.min(index, sortedColumn.length - 1)];
    }

    private float calculatePercentile(int[] column, float percentile) {
        int[] sortedColumn = Arrays.copyOf(column, column.length); 
		Arrays.sort(sortedColumn); 
        int index = (int) Math.ceil(percentile / 100.0 * (sortedColumn.length - 1));
        return (float)sortedColumn[Math.min(index, sortedColumn.length - 1)];
    }

    public int[] transform(float[] column) {
		int[] transformedArray = new int [column.length];
        for (int i = 0; i < column.length; i++) {
            transformedArray[i] = (transformValue(column[i], binEdges) + 1);
        }
        return transformedArray;
    }

    public int[] transform(int[] column) {
        float[] columnAsFloat = new float[column.length];
        for (int i = 0; i < column.length; i++) {
            columnAsFloat[i] = (float)column[i];
        }
        return transform(columnAsFloat);
    }

    private int transformValue(float value, float[] edges) {
        for (int i = 0; i < edges.length - 1; i++) {
			if (value==-999) return -999;//missing value case, return value as-is 
            if (value >= edges[i] && value < edges[i + 1]) {
                return i;
            }
        }
        return (edges.length - 1);
    }

    public int[] fitTransform(float[] column) {
        fit(column);
        return transform(column);
    }

    public int[] fitTransform(int[] column) {
        fit(column);
        return transform(column);
    }

    public float[] getBinEdges() {
        return binEdges;
    }
}
