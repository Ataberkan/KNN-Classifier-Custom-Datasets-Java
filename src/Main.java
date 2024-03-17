import java.io.*;
import java.util.*;

public class Main {
    private List<List<Double>> trainData;
    private int k;

    public Main(String trainFilePath) throws IOException {
        this.trainData = loadDataset(trainFilePath);
    }

    private List<List<Double>> loadDataset(String filepath) throws IOException {
        List<List<Double>> dataset = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            List<Double> row = new ArrayList<>();
            for (int i = 0; i < parts.length - 1; i++) {
                row.add(Double.parseDouble(parts[i]));
            }
            row.add((double)parts[parts.length - 1].hashCode());
            dataset.add(row);
        }
        br.close();
        return dataset;
    }

    public double euclideanDistanceSquared(List<Double> row1, List<Double> row2) {
        double distance = 0.0;
        for (int i = 0; i < row1.size() - 1; i++) {
            distance += Math.pow(row1.get(i) - row2.get(i), 2);
        }
        return distance;
    }

    public List<List<Double>> getNeighbors(List<Double> testRow) {
        List<List<Double>> neighbors = new ArrayList<>();
        trainData.stream().sorted((row1, row2) ->
                        Double.compare(euclideanDistanceSquared(testRow, row1), euclideanDistanceSquared(testRow, row2)))
                .limit(k)
                .forEachOrdered(neighbors::add);
        return neighbors;
    }

    public double predictClassification(List<Double> testRow) {
        List<List<Double>> neighbors = getNeighbors(testRow);
        Map<Double, Integer> classVotes = new HashMap<>();
        for (List<Double> neighbor : neighbors) {
            Double response = neighbor.get(neighbor.size() - 1);
            classVotes.put(response, classVotes.getOrDefault(response, 0) + 1);
        }
        return Collections.max(classVotes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public void setK(int k) {
        this.k = k;
    }

    public void classifyTestSet(String testFilePath) throws IOException {
        List<List<Double>> testData = loadDataset(testFilePath);
        int correct = 0;
        for (List<Double> testRow : testData) {
            double output = predictClassification(testRow);
            if (testRow.get(testRow.size() - 1).equals(output)) {
                correct++;
            }
            System.out.println("Expected=" + testRow.get(testRow.size() - 1) + ", Predicted=" + output);
        }
        System.out.println("Accuracy: " + ((double) correct / testData.size()));
    }

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter path to the training file: ");
        String trainFilePath = scanner.nextLine();
        Main knn = new Main(trainFilePath);

        System.out.println("Enter the number K (number of nearest neighbors): ");
        int k = scanner.nextInt();
        knn.setK(k);
        scanner.nextLine();

        while (true) {
            System.out.println("\nChoose an option:\na) Classify all observations from the test set\nb) Classify an observation given in the console\nc) Change k\nd) Exit");
            String option = scanner.nextLine();

            switch (option) {
                case "a":
                    System.out.println("Enter path to the test file: ");
                    String testFilePath = scanner.nextLine();
                    knn.classifyTestSet(testFilePath);
                    break;
                case "b":
                    System.out.println("Enter an observation separated by commas (without the label): ");
                    String[] parts = scanner.nextLine().split(",");
                    List<Double> observation = new ArrayList<>();
                    for (String part : parts) {
                        observation.add(Double.parseDouble(part));
                    }
                    observation.add(-1.0);
                    double predictedLabel = knn.predictClassification(observation);
                    System.out.println("Predicted label: " + predictedLabel);
                    break;
                case "c":
                    System.out.println("Enter the new value of K: ");
                    k = scanner.nextInt();
                    knn.setK(k);
                    scanner.nextLine();
                    break;
                case "d":
                    return;
            }
        }
    }
}
