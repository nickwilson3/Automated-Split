// Prevent duplicate rankings in the index page
document.addEventListener('DOMContentLoaded', function() {
    // Get all rank select elements
    const rankSelects = document.querySelectorAll('.rank-select');
    
    // Add change event listeners to each select
    rankSelects.forEach(select => {
        select.addEventListener('change', function() {
            // Get the selected value
            const selectedValue = this.value;
            const currentSelectId = this.id;
            
            // Check if any other select has the same value
            rankSelects.forEach(otherSelect => {
                if (otherSelect.id !== currentSelectId && otherSelect.value === selectedValue) {
                    // Find a value that's not used
                    let newValue = '1';
                    const usedValues = Array.from(rankSelects).map(s => s.value);
                    
                    for (let i = 1; i <= 4; i++) {
                        const val = i.toString();
                        if (!usedValues.includes(val) || val === selectedValue) {
                            newValue = val;
                            break;
                        }
                    }
                    
                    // Set the other select to the new value
                    otherSelect.value = newValue;
                }
            });
        });
    });
});