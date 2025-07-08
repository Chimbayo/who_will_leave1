document.addEventListener("DOMContentLoaded", function () {
    fetch("/api/customers?page=1&size=10000")
        .then(response => response.json())
        .then(data => {
            const customers = data.customers || [];
            const tbody = document.getElementById("results-body");
            tbody.innerHTML = "";

            customers.forEach(customer => {
                const prob = customer.Churn_Probability !== null
                    ? (customer.Churn_Probability * 100).toFixed(2) + "%"
                    : "N/A";

                const predictionLabel =
                    customer.prediction === 1
                        ? `<span class="text-red-600">Customer will leave</span>`
                        : customer.prediction === 0
                            ? `<span class="text-green-600">Customer will stay</span>`
                            : "Unknown";

                const row = `
                    <tr class="border-b hover:bg-gray-50">
                        <td class="p-3">${customer.Customer_ID}</td>
                        <td class="p-3">${customer.Age}</td>
                        <td class="p-3">${customer.Gender}</td>
                        <td class="p-3">${customer.District}</td>
                        <td class="p-3">${customer.Region}</td>
                        <td class="p-3">${customer['Location_Type']}</td>
                        <td class="p-3">${customer['Customer_Type']}</td>
                        <td class="p-3">${customer['Employment_Status']}</td>
                        <td class="p-3">${customer['Income_Level']}</td>
                        <td class="p-3">${customer['Education_Level']}</td>
                        <td class="p-3">${customer.Tenure}</td>
                        <td class="p-3">${customer.Balance}</td>
                        <td class="p-3">${customer['Credit_Score']}</td>
                        <td class="p-3">${customer['Outstanding_Loans']}</td>
                        <td class="p-3">${customer['Num_Of_Products']}</td>
                        <td class="p-3">${customer['Mobile_Banking_Usage']}</td>
                        <td class="p-3">${customer['Number_of_Transactions_per_Month']}</td>
                        <td class="p-3">${customer['Num_Of_Complaints']}</td>
                        <td class="p-3">${customer['Proximity_to_NearestBranch_or_ATM_km']}</td>
                        <td class="p-3">${customer['Mobile_Network_Quality']}</td>
                        <td class="p-3">${customer['Owns_Mobile_Phone']}</td>
                        <td class="p-3">${predictionLabel}</td>
                        <td class="p-3">${prob}</td>
                    </tr>
                `;
                tbody.insertAdjacentHTML("beforeend", row);
            });

            console.log(`Loaded ${customers.length} customers`);
        })
        .catch(error => {
            console.error("Error loading customer data:", error);
            document.getElementById("results-body").innerHTML = `
                <tr>
                    <td colspan="23" class="text-red-600 p-3">Error loading data.</td>
                </tr>
            `;
        });
});

document.addEventListener("DOMContentLoaded", function () {
    fetch("/api/customers/churn_count")
        .then(response => response.json())
        .then(data => {
            const count = data.churn_count ?? 0;
            document.getElementById("predicted-churn").textContent = count;
        })
        .catch(error => {
            console.error("Error fetching churn count:", error);
            document.getElementById("predicted-churn").textContent = "Error";
        });
});

document.addEventListener("DOMContentLoaded", function () {
    fetch('/api/customers/churn_summary')
        .then(response => response.json())
        .then(data => {
            buildChurnCharts(data);
        })
        .catch(error => {
            console.error("Error loading churn summary:", error);
            document.getElementById("churn-summary-charts").textContent = "Failed to load summary.";
        });
});

function buildChurnCharts(summary) {
    const container = document.getElementById("churn-summary-charts");
    container.innerHTML = "";

    for (const [attr, rows] of Object.entries(summary)) {
        const canvas = document.createElement("canvas");
        canvas.style.marginBottom = "30px";

        const h4 = document.createElement("h4");
        h4.textContent = attr.replace(/_/g, " ");

        container.appendChild(h4);
        container.appendChild(canvas);

        const labels = rows.map(r => String(r.value));
        const values = rows.map(r => parseFloat(r.percentage.toFixed(2)));

        new Chart(canvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "% of churners",
                    data: values,
                    backgroundColor: "rgba(255, 99, 132, 0.6)"
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => context.parsed.y + "%"
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: "% churn" }
                    }
                }
            }
        });
    }
}
