//prevent duplicate rankings in the index page
document.addEventListener('DOMContentLoaded', function() {
    // Get all rank select elements
    const rankSelects = document.querySelectorAll('.rank-select');
    
    // Track the previous values
    const previousValues = {};
    rankSelects.forEach(select => {
        previousValues[select.id] = select.value;
    });
    
    // Add change event listeners to each select
    rankSelects.forEach(select => {
        select.addEventListener('change', function() {
            // Get the selected value
            const selectedValue = this.value;
            const currentSelectId = this.id;
            
            // Look for duplicates
            let duplicateSelect = null;
            rankSelects.forEach(otherSelect => {
                if (otherSelect.id !== currentSelectId && otherSelect.value === selectedValue) {
                    duplicateSelect = otherSelect;
                }
            });
            
            // If we found a duplicate, swap values
            if (duplicateSelect) {
                // Set the other select to the previous value of the current select
                duplicateSelect.value = previousValues[currentSelectId];
                console.log(`Swapped value: ${duplicateSelect.id} now has value ${previousValues[currentSelectId]}`);
            }
            
            // Update the previous value for the current select
            previousValues[currentSelectId] = selectedValue;
            
            // Log current values for debugging
            console.log('Current rank values:');
            rankSelects.forEach(s => {
                console.log(`${s.id}: ${s.value}`);
            });
        });
    });
});