document.addEventListener('DOMContentLoaded', function() {
    const dataScript = document.getElementById('data');
    if (!dataScript) {
        console.error("Data script tag not found. Make sure it's in analytics.html.");
        return;
    }
    const jsonData = JSON.parse(dataScript.textContent);

    function processChartData(apiData) {
        const labels = [];
        const counts = [];
        apiData.forEach(item => {
            labels.push(item._id === null ? "N/A" : String(item._id));
            counts.push(item.count || 0);
        });
        return { labels, counts };
    }

    function processAgeData(ageApiData) {
        const ageLabels = [];
        const ageCounts = [];
        
        const ageRangeMap = {
            0: '0-17',
            18: '18-25',
            26: '26-40',
            41: '41-60',
            61: '61+'
        };

        const numericAgeData = ageApiData
            .filter(item => typeof item._id === 'number')
            .sort((a, b) => a._id - b._id);

        const unknownAgeData = ageApiData.find(item => item._id === "Unknown");

        numericAgeData.forEach(item => {
            const label = ageRangeMap[item._id] || String(item._id);
            ageLabels.push(label);
            ageCounts.push(item.count);
        });

        if (unknownAgeData) {
            ageLabels.push(unknownAgeData._id);
            ageCounts.push(unknownAgeData.count);
        }

        return { labels: ageLabels, counts: ageCounts };
    }

    // --- Initialize and Render Charts (Existing Chart.js components) ---

    // Gender Chart
    const { labels: genderLabels, counts: genderCounts } = processChartData(jsonData.gender_data);
    new Chart(document.getElementById('genderChart'), {
        type: 'bar',
        data: {
            labels: genderLabels,
            datasets: [{
                label: 'Complaints by Gender',
                data: genderCounts,
                backgroundColor: 'rgba(153, 102, 255, 0.7)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Age Group Chart
    const { labels: ageLabels, counts: ageCounts } = processAgeData(jsonData.age_data);
    new Chart(document.getElementById('ageGroupChart'), {
        type: 'bar',
        data: {
            labels: ageLabels,
            datasets: [{
                label: 'Complaints by Age Group',
                data: ageCounts,
                backgroundColor: 'rgba(255, 159, 64, 0.7)',
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Attack Type Chart
    const { labels: attackLabels, counts: attackCounts } = processChartData(jsonData.attack_type_data);
    new Chart(document.getElementById('attackTypeChart'), {
        type: 'pie',
        data: {
            labels: attackLabels,
            datasets: [{
                label: 'Complaints by Type of Attack',
                data: attackCounts,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(255, 205, 86, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)'
                ],
                borderColor: 'rgba(255, 255, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed !== null) {
                                label += context.parsed;
                                if (context.dataset.data) {
                                    let total = context.dataset.data.reduce((acc, val) => acc + val, 0);
                                    let percentage = total > 0 ? (context.parsed / total * 100).toFixed(1) + '%' : '0%';
                                    label += ` (${percentage})`;
                                }
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });


    // === NEW: Google GeoChart for Complaints by State ===
    // Load the Google Charts package and draw the chart when ready
    google.charts.load('current', {'packages':['geochart']});
    google.charts.setOnLoadCallback(drawRegionsMap);

    function drawRegionsMap() {
        // Data format: [['State', 'Complaints'], ['Maharashtra', 27], ...]
        let stateChartData = [['State', 'Complaints']];
        // Ensure that _id from MongoDB matches the state names Google Charts expects
        // It might be 'Maharashtra' or 'IN-MH' - Google charts generally maps by full name well.
        jsonData.state_data.forEach(item => {
            // Using item._id as state name directly. If you used state codes, adjust mapping.
            stateChartData.push([item._id, item.count]);
        });

        const data = google.visualization.arrayToDataTable(stateChartData);

        const options = {
            region: 'IN', // Focus on India
            resolution: 'provinces', // Show states/provinces
            displayMode: 'regions', // Color the regions (states)
            colorAxis: {colors: ['#e0f7fa', '#007ac1']}, // Light to dark blue gradient (can customize)
            enableInteractivity: true,
            datalessRegionColor: '#f8f8f8', // Color for states with no data
            defaultColor: '#f8f8f8',
            backgroundColor: '#f9f9f9' // Match card background
        };

        const chart = new google.visualization.GeoChart(document.getElementById('regions_div'));
        chart.draw(data, options);
    }
    // === END NEW GeoChart ===
});