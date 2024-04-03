$(document).ready(function() {
    $('#file-upload').submit(function(event) {
        event.preventDefault();
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: '/segment',
            type: 'POST',
            data: formData,
            async: true,
            cache: false,
            contentType: false,
            processData: false,
            beforeSend: function() {
                $('#loading-overlay').removeClass('d-none');
            },
            success: function(response) {
                $('#segmented-img').attr('src', 'data:image/png;base64,' + response.image);
                $('#segmented-image').removeClass('d-none');
                $('#loading-overlay').addClass('d-none');
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
                alert('An error occurred while processing the image.');
                $('#loading-overlay').addClass('d-none');
            }
        });
    });
});



