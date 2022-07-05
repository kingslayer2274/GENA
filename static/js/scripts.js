jQuery(document).ready(function ($) {

    //menu
    // $('.menu-button').click(function () {
    //     $(this).toggleClass('is-active');
    //     $('.main-header__bot').toggleClass('mobile-menu');
    //     $('body').toggleClass('overflow');
    // });

    //consultation
    // $('.order-consultation').click(function () {
    //     $('.consultation').addClass('is-active');
    //     $('#main-header').hide();
    //     $('body').addClass('overflow');
    // });
    // $('.close-form').click(function () {
    //     $('.consultation').removeClass('is-active');
    //     $('#main-header').show();
    //     $('body').removeClass('overflow');
    // });

    //progress bar
    $(document).on("scroll", function () {
        var pixels = $(document).scrollTop();
        var pageHeight = $(document).height() - $(window).height();
        var progress = 100 * pixels / pageHeight;

        $("div.progress").css("width", progress + "%");
    })

    //smooth scrolling
    var $root = $('html, body');

    $('a[href^="#"]').click(function () {
        $root.animate({
            scrollTop: $($.attr(this, 'href')).offset().top - $('#main-header').height()
        }, 700);

        return false;
    });

    //map
    ymaps.ready(init);
    function init() {
        var myMap = new ymaps.Map("map", {
            center: [43.241329538297386, 76.92271082209015],
            zoom: 17
        });
        var myPlacemark = new ymaps.Placemark([43.241329538297386, 76.92271082209015]);
        myMap.geoObjects.add(myPlacemark);
        myMap.behaviors.disable('scrollZoom');
    }

    //sliders
    $('.reviews-list').slick({
        arrows: true,
        dots: false,
        slidesToShow: 4,
        autoplay: true,
        autoplayspeed: 2000,
        infinite: true,
        prevArrow: $('.review-slider__buttons .prev-button'),
        nextArrow: $('.review-slider__buttons .next-button'),
        responsive: [
            {
                breakpoint: 1170,
                settings: {
                    slidesToShow: 3,
                    slidesToScroll: 3,
                }
            },
            {
                breakpoint: 768,
                settings: {
                    slidesToShow: 2,
                    slidesToScroll: 2,
                    centerMode: true,
                    centerPadding: '30px',
                }
            },
            {
                breakpoint: 550,
                settings: {
                    infinite: false,
                    slidesToShow: 1,
                    slidesToScroll: 1,
                    centerMode: true,
                    centerPadding: '30px',
                }
            },
        ]
    });

    $('.section-about__imgs').slick({
        arrows: false,
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        mobileFirst: true,
        autoplay: true,
        autoplayspeed: 2000,
        responsive: [
            {
                breakpoint: 768,
                settings: "unslick"
            },
        ]
    })

    //observer
    const links = document.querySelectorAll('.menu-item');
    const sections = document.querySelectorAll('.menu-anchor');
    function changeLinkState() {
        let index = sections.length;

        while (--index && window.scrollY + 150 < sections[index].offsetTop - 250) { }

        links.forEach((link) => link.classList.remove('current-menu-item'));
        links[index].classList.add('current-menu-item');
    }

    changeLinkState();
    window.addEventListener('scroll', changeLinkState);


    //E-mail Ajax Send
    // $(".contact-form").submit(function () {
    //     $form = $(this);
    //     if ($('.check-input').hasClass('valid')) {
    //         var th = $(this);
    //         $.ajax({
    //             type: "POST",
    //             url: "/mail.php",
    //             data: th.serialize()
    //         }).done(function () {
    //             th.trigger("reset");
    //             $form.closest('.section-form').addClass('contact-sent');
    //         });
    //         return false;
    //     } else {
    //         return false;
    //     }
    // });

    // $('.contact-sumbit').click(function () {
    //     if ($(this).siblings('.check-input').find('.contact-method').val().length == 0) {
    //         $(this).next('.contact-error').show();
    //         $(this).siblings('.check-input').addClass('invalid');
    //     }
    // });

    // $(".contact-input input").change(function () {
    //     $val = $(this).val();
    //     if ($val.length > 0) {
    //         $(this).parent().removeClass('invalid').addClass('valid').find('label').hide();
    //     } else {
    //         $(this).parent().removeClass('invalid valid').find('label').show();
    //     }
    // });

    // $(".contact-method").change(function () {
    //     $val = $(this).val();
    //     $rv_check = /^([a-zA-Z0-9_.-])+@([a-zA-Z0-9_.-])+\.([a-zA-Z])+([a-zA-Z])+/;
    //     $check_phone = /^[+-]?\d+$/;
    //     if ($val.length > 3 && $val != '' && ($rv_check.test($val) || $check_phone.test($val))) {
    //         $(this).parent().removeClass('invalid').addClass('valid');
    //         $(this).parent().siblings('.contact-error').hide();
    //     } else if ($val.length == 0) {
    //         $(this).parent().removeClass('invalid valid');
    //     } else {
    //         $(this).parent().removeClass('valid').addClass('invalid');
    //     }
    // });

    //ytPlayer
    $('.video-container').click(function () {
        $(this).addClass('player');
    });
});