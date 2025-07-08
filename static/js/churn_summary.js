document.addEventListener("DOMContentLoaded", function () {
    fetch('/api/customers/churn_summary')
        .then(response => response.json())
        .then(data => {
            if (Object.keys(data).length === 0) {
                document.getElementById("churn-summary-charts").textContent =
                    "No churn data available.";
            } else {
                buildChurnCharts(data);
            }
        })
        .catch(error => {
            console.error("Error fetching churn summary:", error);
            document.getElementById("churn-summary-charts").textContent =
                "Error loading churn summary.";
        });
});

function buildChurnCharts(summary) {
    const container = document.getElementById("churn-summary-charts");
    container.innerHTML = "";

    const downloadAllBtn = document.createElement("button");
    downloadAllBtn.textContent = "Download ALL Graphs as PDF";
    downloadAllBtn.onclick = downloadAllChartsAsPDF;
    downloadAllBtn.style.marginBottom = "20px";
    container.appendChild(downloadAllBtn);

    for (const [attr, rows] of Object.entries(summary)) {
        const section = document.createElement("div");
        section.style.marginBottom = "50px";

        const h4 = document.createElement("h4");
        h4.textContent = attr.replace(/_/g, " ");
        section.appendChild(h4);

        const downloadBtn = document.createElement("button");
        downloadBtn.textContent = "Download CSV";
        downloadBtn.onclick = () => downloadCSV(attr, rows);
        section.appendChild(downloadBtn);

        const imgDownloadBtn = document.createElement("button");
        imgDownloadBtn.textContent = "Download Graph";
        imgDownloadBtn.style.marginLeft = "10px";
        imgDownloadBtn.onclick = () => {
            const link = document.createElement("a");
            link.download = `${attr}_churn_chart.png`;
            link.href = canvas.toDataURL("image/png");
            link.click();
        };
        section.appendChild(imgDownloadBtn);

        const canvas = document.createElement("canvas");
        section.appendChild(canvas);

        container.appendChild(section);

        const labels = rows.map(r => String(r.value));
        const values = rows.map(r => parseFloat(r.percentage));

        new Chart(canvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "% churners",
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
                            label: ctx => `${ctx.parsed.y}%`
                        }
                    },
                    datalabels: {
                        anchor: 'end',
                        align: 'end',
                        formatter: (value) => `${value}%`,
                        color: '#333',
                        font: {
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: "% churn"
                        }
                    }
                }
            },
            plugins: [ChartDataLabels]
        });
    }
}

function downloadCSV(attr, rows) {
    let csv = "Value,Churners,Percentage\n";
    rows.forEach(row => {
        csv += `${row.value},${row.churners},${row.percentage}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `${attr}_churn_summary.csv`;
    a.click();

    window.URL.revokeObjectURL(url);
}

function downloadAllChartsAsPDF() {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF();

    const charts = document.querySelectorAll("canvas");

    let y = 10;

    charts.forEach((canvas, idx) => {
        const imgData = canvas.toDataURL("image/png");
        pdf.addImage(imgData, 'PNG', 10, y, 180, 80);
        y += 90;

        if (y > 250 && idx < charts.length - 1) {
            pdf.addPage();
            y = 10;
        }
    });

    pdf.save("churn_summary.pdf");
}
