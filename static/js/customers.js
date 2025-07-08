
        document.addEventListener("DOMContentLoaded", function () {
            fetch("/api/customers?page=1&size=10000")
                .then(response => response.json())
                .then(data => {
                    const customers = data.customers || [];
                    const tbody = document.getElementById("results-body");
                    tbody.innerHTML = "";

                    customers.forEach(customer => {
                        const prob = customer.churn_probability !== null
                            ? (customer.churn_probability * 100).toFixed(2) + "%"
                            : "N/A";

                        const predictionLabel =
                            customer.prediction === 1
                                ? `<span class="text-red-600">Customer will leave</span>`
                                : customer.prediction === 0
                                    ? `<span class="text-green-600">Customer will stay</span>`
                                    : "Unknown";

                        const row = `
                            <tr class="border-b hover:bg-gray-50">
                                <td class="p-3">${customer.customer_id ?? ""}</td>
                                <td class="p-3">${customer.age ?? ""}</td>
                                <td class="p-3">${customer.gender ?? ""}</td>
                                <td class="p-3">${customer.district ?? ""}</td>
                                <td class="p-3">${customer.region ?? ""}</td>
                                <td class="p-3">${customer.location_type ?? ""}</td>
                                <td class="p-3">${customer.customer_type ?? ""}</td>
                                <td class="p-3">${customer.employment_status ?? ""}</td>
                                <td class="p-3">${customer.income_level ?? ""}</td>
                                <td class="p-3">${customer.education_level ?? ""}</td>
                                <td class="p-3">${customer.tenure ?? ""}</td>
                                <td class="p-3">${customer.balance ?? ""}</td>
                                <td class="p-3">${customer.credit_score ?? ""}</td>
                                <td class="p-3">${customer.outstanding_loans ?? ""}</td>
                                <td class="p-3">${customer.num_of_products ?? ""}</td>
                                <td class="p-3">${customer.mobile_banking_usage ?? ""}</td>
                                <td class="p-3">${customer.number_of_transactions_per_month ?? ""}</td>
                                <td class="p-3">${customer.num_of_complaints ?? ""}</td>
                                <td class="p-3">${customer.proximity_to_nearestbranch_or_atm_km ?? ""}</td>
                                <td class="p-3">${customer.mobile_network_quality ?? ""}</td>
                                <td class="p-3">${customer.owns_mobile_phone ?? ""}</td>
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
    