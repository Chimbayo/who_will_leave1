
document.addEventListener('DOMContentLoaded', function () {
    const slider = document.getElementById('probability-slider');
    const valueSpan = document.getElementById('probability-value');
    const vapan = document.getElementById('threshold-display');
    slider.addEventListener('input', function () {
        valueSpan.textContent = slider.value + '%';
        vapan.textContent = slider.value + '%';
    });
});

document.addEventListener("DOMContentLoaded", function () {
    const slider = document.getElementById("probability-slider");
    const probabilityValue = document.getElementById("probability-value");
    const totalCountDisplay = document.getElementById("total-count-display");
    const exportBtn = document.getElementById("export-btn");
    const resultsBody = document.getElementById("results-body");

    function loadCustomers(threshold) {
    fetch(`/api/customers/predicted?threshold=${threshold}`)
        .then(response => response.json())
        .then(data => {
        resultsBody.innerHTML = "";

        totalCountDisplay.textContent = `Total customers with ≥ ${threshold}% probability of leaving: ${data.length}`;
        probabilityValue.textContent = `${threshold}%`;

        if (data.length === 0) {
            resultsBody.innerHTML = `
            <tr>
                <td colspan="23" style="text-align:center; color: #666;">
                No customers found above this probability.
                </td>
            </tr>`;
            return;
        }

        for (const customer of data) {
            const row = document.createElement("tr");

            row.innerHTML = `
            <td>${customer.Customer_ID}</td>
            <td>${customer.Age}</td>
            <td>${customer.Gender}</td>
            <td>${customer.District}</td>
            <td>${customer.Region}</td>
            <td>${customer.Location_Type}</td>
            <td>${customer.Customer_Type}</td>
            <td>${customer.Employment_Status}</td>
            <td>${customer.Income_Level}</td>
            <td>${customer.Education_Level}</td>
            <td>${customer.Tenure}</td>
            <td>${customer.Balance}</td>
            <td>${customer.Credit_Score}</td>
            <td>${customer.Outstanding_Loans}</td>
            <td>${customer.Num_Of_Products}</td>
            <td>${customer.Mobile_Banking_Usage}</td>
            <td>${customer.Number_of_Transactions_per_Month}</td>
            <td>${customer.Num_Of_Complaints}</td>
            <td>${customer.Proximity_to_NearestBranch_or_ATM_km}</td>
            <td>${customer.Mobile_Network_Quality}</td>
            <td>${customer.Owns_Mobile_Phone}</td>
            <td>${customer.prediction}</td>
            <td>${(parseFloat(customer.Churn_Probability) * 100).toFixed(2)}%</td>
            `;
            resultsBody.appendChild(row);
        }
        })
        .catch(error => {
        console.error("Error fetching customers:", error);
        });
    }

    // Initial load
    loadCustomers(slider.value);

    slider.addEventListener("input", () => {
    loadCustomers(slider.value);
    });

    exportBtn.addEventListener("click", () => {
    const threshold = slider.value;
    let csvContent = "";

    csvContent += `"Customers with ≥ ${threshold}% Probability of Leaving"\n\n`;

    // Get table headers
    const headers = [];
    document.querySelectorAll("#results-table thead th").forEach(th => {
        headers.push(`"${th.textContent.trim()}"`);
    });
    csvContent += headers.join(",") + "\n";

    // Get table rows
    const rows = [];
    resultsBody.querySelectorAll("tr").forEach(tr => {
        const cells = [];
        tr.querySelectorAll("td").forEach(td => {
        let text = td.textContent.trim();
        text = text.replace(/"/g, '""'); // Escape quotes
        cells.push(`"${text}"`);
        });
        rows.push(cells.join(","));
    });

    if (rows.length === 0) {
        alert("No data to export.");
        return;
    }

    csvContent += rows.join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `customers_prob_${threshold}.csv`);
    link.click();
    URL.revokeObjectURL(url);
    });
});

document.addEventListener("DOMContentLoaded", function () {
    const entryTypeSelect = document.getElementById("entry-type");
    const singleEntryFields = document.getElementById("single-entry-fields");
    const multipleEntryFields = document.getElementById("multiple-entry-fields");

    entryTypeSelect.addEventListener("change", function () {
        if (this.value === "single") {
            singleEntryFields.style.display = "block";
            multipleEntryFields.style.display = "none";
        } else if (this.value === "multiple") {
            singleEntryFields.style.display = "none";
            multipleEntryFields.style.display = "block";
        }
    });

    // Handle CSV upload
    const fileInput = document.getElementById("batch-upload");
    fileInput.addEventListener("change", function () {
        const file = this.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvText = e.target.result;
            processCsv(csvText);
        };
        reader.readAsText(file);
    });

    function processCsv(csvText) {
        // Parse CSV text → array of objects
        const rows = csvText.trim().split("\n");
        const headers = rows.shift().split(",").map(h => h.trim());
        const customers = rows.map(line => {
            const values = line.split(",");
            const obj = {};
            headers.forEach((h, i) => {
                obj[h] = values[i] !== undefined ? values[i].trim() : "";
            });
            return obj;
        });

        console.log("Parsed CSV:", customers);

        // Send all customers to backend
        fetch("/predict_batch", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(customers)
        })
        .then(res => res.json())
        .then(response => {
            alert(`Batch upload complete. ${response.processed} customers added!`);
        })
        .catch(err => {
            console.error(err);
            alert("Batch upload failed.");
        });
    }
});

