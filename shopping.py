import csv
import sys


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            administrative = int(row[0])
            administrative_duration = float(row[1])
            informational = int(row[2])
            informational_duration = float(row[3])
            product_related = int(row[4])
            product_related_duration = float(row[5])
            bounce_rates = float(row[6])
            exit_rates = float(row[7])
            page_values = float(row[8])
            special_day = float(row[9])
            month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul',
                     'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(row[10])
            operating_systems = int(row[11])
            browser = int(row[12])
            region = int(row[13])
            traffic_type = int(row[14])
            visitor_type = int(row[15] == 'Returning_Visitor')
            weekend = 1 if row[16] == 'TRUE' else 0
            evidence.append([
                administrative, administrative_duration, informational, informational_duration,
                product_related, product_related_duration, bounce_rates, exit_rates, page_values,
                special_day, month, operating_systems, browser, region, traffic_type, visitor_type, weekend
            ])

            labels.append(1 if row[-1] == 'TRUE' else 0)
    return (evidence, labels)


def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):

    # Compute how well we performed
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
    sensitivty = float(tp/(tp+fn))
    specificty = float(tn/(tn+fp))
    return (sensitivty, specificty)


if __name__ == "__main__":
    main()
